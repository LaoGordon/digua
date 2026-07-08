"""
generate_field_dataset.py — 生成代价感知势场训练数据。

与 generate_dataset.py (路径掩码目标) 的区别:
  - 监督信号: cost_aware_dijkstra_field (dense, smooth, [0,1])
  - 不是 A* 路径掩码 (sparse, binary)
  - 模型学的是"代价感知的 cost-to-go 势场", 不是路径图

用法:
  python3 generate_field_dataset.py --n-samples 8000 --out-dir ./dataset_field --model-size 256

输出:
  train.npz / val.npz
    x: (N, 4, M, M) float32  — ch0=障碍 ch1=起点热图 ch2=终点热图 ch3=膨胀半径
    y: (N, 1, M, M) float32  — cost_aware_dijkstra_field (代价感知势场)
    meta.npz: Nav2参数 + alpha + mode + model_size
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from synth_maps import generate_mixed
from nav2_inflation import inflate_nav2
from planner_utils import cost_aware_dijkstra_field, gaussian_peak


def gen_sample(size, rng, nav_params, alpha, mode, min_dist_cells=30, max_tries=40):
    """生成一个训练样本。

    返回 (input_4ch, field_target) 或 None。
    """
    grid, style = generate_mixed(size, size, rng)
    cost, hard = inflate_nav2(
        grid,
        nav_params["robot_radius"],
        nav_params["inflation_radius"],
        nav_params["cost_scaling"],
        nav_params["resolution"],
    )
    obs = hard.astype(np.float32)

    # 在硬障碍外的自由区采样起终点
    free_cells = np.argwhere(~hard)
    if len(free_cells) < min_dist_cells:
        return None

    for _ in range(max_tries):
        si = rng.integers(len(free_cells))
        gi = rng.integers(len(free_cells))
        s = tuple(free_cells[si])
        g = tuple(free_cells[gi])
        if abs(s[0] - g[0]) + abs(s[1] - g[1]) < min_dist_cells:
            continue

        # 代价感知 Dijkstra 势场 (以 goal 为源)
        field = cost_aware_dijkstra_field(hard, cost, g, alpha=alpha, mode=mode)
        # 检查 field 是否有效 (不应全 NaN)
        valid = ~np.isnan(field)
        if valid.sum() < min_dist_cells:
            continue

        # 构造输入通道
        start_heat = gaussian_peak((size, size), s, sigma=3.0)
        goal_heat = gaussian_peak((size, size), g, sigma=3.0)
        inflation_range_cells = (nav_params["robot_radius"] + nav_params["inflation_radius"]) \
                                / nav_params["resolution"]
        infl_map = np.full((size, size), inflation_range_cells / size, dtype=np.float32)

        inp = np.stack([obs, start_heat, goal_heat, infl_map], axis=0)
        return inp, field[None, ...], style

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=8000)
    ap.add_argument("--out-dir", default="./dataset_field")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-size", type=int, default=256)
    ap.add_argument("--min-dist-cells", type=int, default=30)
    # Nav2 参数
    ap.add_argument("--robot-radius", type=float, default=0.20)
    ap.add_argument("--inflation-radius", type=float, default=0.25)
    ap.add_argument("--cost-scaling", type=float, default=3.0)
    ap.add_argument("--resolution", type=float, default=0.05)
    # 代价感知参数
    ap.add_argument("--alpha", type=float, default=20.0,
                    help="代价感知权重 (越大越排斥贴墙, add mode 推荐 20)")
    ap.add_argument("--mode", default="add",
                    choices=["linear", "exp", "add"],
                    help="代价模式: linear/exp/add")
    ap.add_argument("--synth-size", type=int, default=256)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    nav_params = {
        "robot_radius": args.robot_radius,
        "inflation_radius": args.inflation_radius,
        "cost_scaling": args.cost_scaling,
        "resolution": args.resolution,
    }

    print(f"[gen_field] Nav2: {nav_params}")
    print(f"[gen_field] Cost: mode={args.mode} alpha={args.alpha}")
    inscribed = args.robot_radius / args.resolution
    infl_range = inscribed + args.inflation_radius / args.resolution
    print(f"[gen_field] inscribed={inscribed:.1f} cells, total_inflation={infl_range:.1f} cells")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    M = args.model_size
    inputs = np.zeros((args.n_samples, 4, M, M), dtype=np.float32)
    targets = np.zeros((args.n_samples, 1, M, M), dtype=np.float32)

    ok = 0
    style_count = {"maze": 0, "obstacles": 0, "corridors": 0}
    total_tries = 0
    while ok < args.n_samples:
        result = gen_sample(args.synth_size, rng, nav_params,
                            alpha=args.alpha, mode=args.mode,
                            min_dist_cells=args.min_dist_cells)
        total_tries += 1
        if result is None:
            continue
        inp_full, tgt_full, style = result
        style_count[style] += 1

        # resize 到 model_size
        if args.synth_size != M:
            def to_M(a):
                return np.asarray(
                    Image.fromarray(a).resize((M, M), Image.BILINEAR),
                    dtype=np.float32)

            inp = np.stack([
                to_M(inp_full[0]), to_M(inp_full[1]),
                to_M(inp_full[2]), to_M(inp_full[3])
            ], axis=0)
            tgt = to_M(tgt_full[0])[None, ...]
        else:
            inp = inp_full
            tgt = tgt_full

        inputs[ok] = inp
        targets[ok] = tgt
        ok += 1
        if ok % 500 == 0:
            print(f"[gen_field] {ok}/{args.n_samples}  "
                  f"风格:{style_count}  "
                  f"field_range=[{tgt.min():.4f},{tgt.max():.4f}]")

    inputs = inputs[:ok]
    targets = targets[:ok]
    print(f"[gen_field] 总尝试 {total_tries} 次, 成功 {ok} 个 ({ok/total_tries*100:.1f}%)")

    # train/val 8:2
    n = len(inputs)
    perm = rng.permutation(n)
    n_val = max(1, n // 5)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    np.savez_compressed(out_dir / "train.npz",
                        x=inputs[train_idx], y=targets[train_idx])
    np.savez_compressed(out_dir / "val.npz",
                        x=inputs[val_idx], y=targets[val_idx])
    # meta
    np.savez(out_dir / "meta.npz", **nav_params,
             alpha=args.alpha, mode=args.mode,
             model_size=M, n_samples=ok)

    print(f"\n[gen_field] 完成: {ok} 样本")
    print(f"[gen_field] 风格: {style_count}")
    print(f"[gen_field] train={len(train_idx)} val={len(val_idx)}")
    print(f"[gen_field] 输出: {out_dir}")
    # 统计 field 的 NaN 比例
    nan_ratio = np.isnan(targets).mean()
    print(f"[gen_field] field NaN 比例: {nan_ratio*100:.2f}%")


if __name__ == "__main__":
    main()
