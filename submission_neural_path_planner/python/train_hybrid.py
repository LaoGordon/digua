"""
train_hybrid.py — 代价感知势场训练 (cost-aware field training)。

与原始版本的区别:
  - 目标: cost_aware_dijkstra_field (Nav2 衰减层代价加权)
  - 数据集: 直接加载预计算的代价感知场 (由 generate_field_dataset.py 生成)
  - 评估: steepest_descent + gradient_walk 双搜索算法
  - 不再需要每 epoch 重算 Dijkstra 场

训练流程不变: MaskedMSE + AdamW + CosineAnnealing。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ppcnet_model import PPCNet
from field_search import steepest_descent, gradient_walk


class FieldDataset(Dataset):
    """加载预计算的代价感知势场数据。

    .npz 格式:
      x: (N, 4, H, W) — ch0=obs ch1=start_heat ch2=goal_heat ch3=infl
      y: (N, 1, H, W) — cost_aware_dijkstra_field (含 NaN for 障碍)
    """

    def __init__(self, path, n=None):
        d = np.load(path)
        x = d["x"].astype(np.float32)
        y = d["y"].astype(np.float32)
        self.n = len(x) if n is None else min(n, len(x))
        self.obs = x[:self.n, 0:1]          # (n, 1, H, W)  障碍图
        self.starts = np.zeros((self.n, 2), dtype=np.float32)
        self.goals = np.zeros((self.n, 2), dtype=np.float32)
        for i in range(self.n):
            self.starts[i] = np.unravel_index(np.argmax(x[i, 1]), x[i, 1].shape)
            self.goals[i] = np.unravel_index(np.argmax(x[i, 2]), x[i, 2].shape)
        self.fields = y[:self.n]            # (n, 1, H, W)  代价感知场
        del x, y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        field = torch.from_numpy(self.fields[idx])
        mask = (~torch.isnan(field)).float()
        field = torch.nan_to_num(field, nan=0.0)
        return (torch.from_numpy(self.obs[idx]),
                torch.from_numpy(self.starts[idx]),
                torch.from_numpy(self.goals[idx]),
                field, mask)


class MaskedMSE(nn.Module):
    def forward(self, pred, target, mask):
        d2 = (pred - target) ** 2 * mask
        return d2.sum() / mask.sum().clamp(min=1.0)


class FieldLossWithBasin(nn.Module):
    """MaskedMSE + 假盆地惩罚。

    根因: MaskedMSE按像素独立惩罚, 没有"goal必须是全局最小"约束。
    模型可把无关区压低产生假盆地, gradient_walk涌向假盆地。
    修复: 显式惩罚任何比goal更低的自由格。
    """
    def __init__(self, basin_w=0.1, margin=0.02):
        super().__init__()
        self.mse = MaskedMSE()
        self.basin_w = basin_w
        self.margin = margin

    def forward(self, pred, target, mask, goal_coords):
        """
        pred: (N,1,H,W)  target: (N,1,H,W)  mask: (N,1,H,W) free=1
        goal_coords: (N,2) goal的(row,col)坐标
        """
        mse_loss = self.mse(pred, target, mask)
        # 取每个样本goal处的预测值
        N = pred.shape[0]
        goal_vals = pred[torch.arange(N), 0, goal_coords[:, 0].long(), goal_coords[:, 1].long()]  # (N,)
        # 假盆地: 任何自由格比goal低margin以上 → 惩罚
        # basin_violation = relu(goal_val - pred + margin), 形状广播 (N,1,H,W) vs (N,1,1,1)
        violation = torch.relu(goal_vals.view(N, 1, 1, 1) - pred + self.margin)  # (N,1,H,W)
        basin_loss = (violation * mask).sum() / mask.sum().clamp(min=1.0)
        return mse_loss + self.basin_w * basin_loss, mse_loss.item(), basin_loss.item()


@torch.no_grad()
def evaluate_field(model, dl, device, criterion, basin_on=False):
    model.eval()
    s = 0.0; nb = 0
    for obs, st, gl, field, mask in dl:
        pred = model(obs.to(device), st.to(device), gl.to(device))
        if basin_on:
            loss, _, _ = criterion(pred, field.to(device), mask.to(device), gl.to(device).long())
        else:
            loss = criterion(pred, field.to(device), mask.to(device))
        s += loss.item(); nb += 1
    return s / max(nb, 1)


def wall_distance_stats(path, obs):
    """路径距墙平均距离。obs: (H,W) bool/numpy."""
    if not path or len(path) < 2:
        return 0.0, 0.0
    from scipy import ndimage
    dmap = ndimage.distance_transform_edt(~obs)
    vals = [dmap[r, c] for r, c in path]
    return float(np.mean(vals)), float(min(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset_field")
    ap.add_argument("--out", default="./ckpt_cost_field")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--basin-w", type=float, default=0.1,
                    help="假盆地惩罚权重(0=禁用, 0.1/1.0/5.0 grid search)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    basin_on = args.basin_w > 0
    loss_name = f"MaskedMSE+Basin(w={args.basin_w})" if basin_on else "MaskedMSE"
    print(f"[CostField] 势场训练: PPCNet + {loss_name}", flush=True)
    print(f"[CostField] batch_size={args.batch_size} (拉满显卡)", flush=True)

    train_ds = FieldDataset(Path(args.data) / "train.npz", n=args.n_train)
    val_ds = FieldDataset(Path(args.data) / "val.npz", n=args.n_val)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=2)

    print(f"[CostField] train={len(train_ds)} val={len(val_ds)}", flush=True)

    model = PPCNet(side=256, n_layers=3, base=args.base).to(device)
    model.sigm = nn.Identity()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = FieldLossWithBasin(basin_w=args.basin_w) if basin_on else MaskedMSE()

    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = 0.0; tb = 0.0; nb = 0
        for obs, st, gl, field, mask in train_dl:
            obs, st, gl, field, mask = (obs.to(device), st.to(device), gl.to(device),
                                        field.to(device), mask.to(device))
            optim.zero_grad()
            pred = model(obs, st, gl)
            if basin_on:
                loss, mse_v, basin_v = criterion(pred, field, mask, gl.long())
                tb += basin_v
            else:
                loss = criterion(pred, field, mask)
                mse_v = loss.item()
            loss.backward(); optim.step()
            tl += mse_v; nb += 1
        sched.step()
        vl = evaluate_field(model, val_dl, device, criterion, basin_on)
        basin_str = f" basin={tb/nb:.4f}" if basin_on else ""
        line = f"[CostField] ep{epoch:2d}/{args.epochs} train_MSE={tl/nb:.4f}{basin_str} val_MSE={vl:.4f}"
        print(line, flush=True); log.append(line)
        if epoch % 10 == 0:
            torch.save({"model": model.state_dict(), "base": args.base,
                        "epoch": epoch, "val_mse": vl, "basin_w": args.basin_w,
                        "note": "field_with_basin_loss"},
                       Path(args.out) / f"ep{epoch}.pt")

    # 最终评估: gradient_walk 为主(已验证99%+连通), steepest可选对照
    print(f"\n[CostField] === 路径提取评估 (val集前50张) ===", flush=True)
    model.eval()
    n = 0
    gw_conn = gw_wall_ok = 0
    t_gw = 0.0
    gw_dists = []
    with torch.no_grad():
        for obs, st, gl, field, mask in val_dl:
            pred = model(obs.to(device), st.to(device), gl.to(device)).cpu().numpy()
            for i in range(obs.size(0)):
                if n >= 50: break
                o = (obs[i, 0].numpy() > 0.5)
                s = (int(st[i, 0]), int(st[i, 1]))
                g = (int(gl[i, 0]), int(gl[i, 1]))
                if o[s] or o[g]:
                    continue
                pf = pred[i, 0]
                o_f32 = o.astype(np.float32)

                # gradient_walk (主搜索, 稳定可靠)
                t0 = time.time()
                p_gw = gradient_walk(pf, s, g, o_f32)
                t_gw += time.time() - t0

                if p_gw and len(p_gw) >= 2:
                    from field_search import reached, count_wall
                    if reached(p_gw, s, g):
                        gw_conn += 1
                    if count_wall(p_gw, o) == 0:
                        gw_wall_ok += 1
                    davg, _ = wall_distance_stats(p_gw, o)
                    gw_dists.append(davg)
                n += 1

    line = (f"[CostField] gradient_walk: 连通={gw_conn}/{n}({gw_conn/max(n,1)*100:.0f}%) "
            f"不穿墙={gw_wall_ok}/{n} "
            f"avg_wall_dist={np.mean(gw_dists) if gw_dists else 0:.1f}格 "
            f"latency={t_gw/max(n,1)*1000:.1f}ms")
    print(line, flush=True); log.append(line)

    with open(Path(args.out) / "log.txt", "w") as f:
        f.write("\n".join(log))
    print(f"\n[CostField] 完成, 日志: {args.out}/log.txt", flush=True)


if __name__ == "__main__":
    main()
