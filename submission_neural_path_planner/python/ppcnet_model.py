"""
ppcnet_model.py — 复刻 Caffagni 的 PPCNet (Path Planning CNN) 架构。

这是"复现验证"用 —— 把已验证可用的 PPCNet 设计套到我们的数据上,
看连通性问题是否解决。

复刻 Caffagni(dcaffo98/path-planning-cnn)的三个关键设计:
  1. GaussianRelativePE: 每个像素得到"到起点/终点的相对距离"连续编码
     → 解决 CNN 感受野局限, 让中间像素知道自己的全局位置
  2. 残差跳跃连接(x = x + skip), 不是 U-Net 的 concat
  3. base=64 的宽网络(原 U-Net 用 base=16 太窄)

损失用 MSE(回归 score map), 不膨胀路径 —— 复刻 Caffagni 的任务定义。

输入适配: 我们的 dataset 存的是高斯热图通道, 这里从热图峰值提取坐标,
在模型内部生成 PE。这样不用重新生成数据。
"""

import math
import torch
import torch.nn as nn


class GaussianRelativePE(nn.Module):
    """高斯相对位置编码。

    给定一个中心点(起点或终点), 为每个像素生成一个高斯值:
      pe[r,c] = alpha * exp(-((r-center_r)^2 + (c-center_c)^2) / (2*sigma^2))
    然后归一化到 [0,1]。

    这样每个像素都"知道"自己离中心点多远 —— 给 CNN 全局坐标信号。
    复刻自 Caffagni model/pe.py。
    """

    def __init__(self, side, sigma=None):
        super().__init__()
        if sigma is None:
            sigma = side / 5.0
        self.sigma_square = sigma ** 2
        self.alpha = 1.0 / (2 * math.pi * self.sigma_square)
        self.side = side
        coord_r = torch.stack([torch.arange(side) for _ in range(side)])
        coord_c = coord_r.T
        self.register_buffer("coord_r", coord_r.float())
        self.register_buffer("coord_c", coord_c.float())

    def forward(self, center):
        """center: (N, 2) 坐标。返回 (N, 1, side, side) 的 PE。"""
        cr = self.coord_r.view(1, self.side, self.side)
        cc = self.coord_c.view(1, self.side, self.side)
        pe = self.alpha * torch.exp(
            -((cr - center[:, 0:1].unsqueeze(1)) ** 2
              + (cc - center[:, 1:2].unsqueeze(1)) ** 2)
            / (2 * self.sigma_square)
        )
        pe = pe / pe.amax(dim=(-1, -2)).view(-1, 1, 1)
        return pe.unsqueeze(1)  # (N, 1, H, W)


class _ConvBlock(nn.Module):
    """PPCNet 的卷积块: conv-act-bn × 2 + 下采样conv。

    复刻 Caffagni ppcnet.py 的 _conv_block, 但加 padding=1 保证尺寸不变:
      下采样: Conv(pad=1,stride=2) 通道翻倍
      上采样: ConvTranspose(pad=1,stride=2,output_padding=1) 通道减半, 尺寸×2
    """

    def __init__(self, in_ch, out_ch, transpose=False, last_output_pad=1):
        super().__init__()
        self.activation = nn.ReLU()
        if transpose:
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.bn3 = nn.BatchNorm2d(out_ch // 2)
            # kernel=3 pad=1 stride=1 → 尺寸不变
            self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
            self.conv2 = nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1)
            # stride=2 pad=1 output_pad=1 → 尺寸精确×2
            self.conv3 = nn.ConvTranspose2d(out_ch, out_ch // 2, 3,
                                            stride=2, padding=1,
                                            output_padding=last_output_pad)
        else:
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.bn3 = nn.BatchNorm2d(out_ch * 2)
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            # stride=2 pad=1 → 尺寸精确÷2
            self.conv3 = nn.Conv2d(out_ch, out_ch * 2, 3, stride=2, padding=1)
        self._fw = nn.Sequential(
            self.conv1, self.activation, self.bn1,
            self.conv2, self.activation, self.bn2,
            self.conv3, self.activation, self.bn3)

    def forward(self, x):
        return self._fw(x)


class PPCNet(nn.Module):
    """Path Planning CNN —— 复刻 Caffagni 架构。

    输入: (N, 1, H, W) 障碍图 + start(N,2) + goal(N,2) 坐标
    输出: (N, 1, H, W) 路径 score map (Sigmoid)
    """

    def __init__(self, side=256, n_layers=3, base=64):
        super().__init__()
        self.pe = GaussianRelativePE(side)
        self.sigm = nn.Sigmoid()

        n_channels = [base * (2 ** i) for i in range(n_layers)]
        # 下采样: 第一层输入 3 通道 (障碍 + pe_start + pe_goal)
        self.conv_down = nn.ModuleList([
            _ConvBlock(c if i > 0 else 3, c)
            for i, c in enumerate(n_channels)
        ])
        self.conv_up = nn.ModuleList([
            _ConvBlock(2 * c, 2 * c, transpose=True, last_output_pad=1)
            for i, c in enumerate(n_channels[::-1])
        ])
        self.bottleneck = nn.Conv2d(base, 1, 3, padding=1)
        self.conv_out = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, obs, start, goal):
        """
        obs:   (N, 1, H, W) 障碍图
        start: (N, 2) 起点坐标
        goal:  (N, 2) 终点坐标
        """
        pe_start = self.pe(start)   # (N, 1, H, W)
        pe_goal = self.pe(goal)     # (N, 1, H, W)
        x = torch.cat([obs, pe_start, pe_goal], dim=1)  # (N, 3, H, W)

        skip_conn = []
        for i, conv in enumerate(self.conv_down):
            x = conv(x)
            if i < len(self.conv_down) - 1:
                skip_conn.append(x)
        for i, conv in enumerate(self.conv_up):
            x = conv(x)
            if i < len(skip_conn):
                skip = skip_conn[-1 - i]
                # 对齐尺寸(odd 尺寸下采样后上采样可能差1格)
                if x.shape[-1] != skip.shape[-1]:
                    import torch.nn.functional as Fn
                    skip = Fn.interpolate(skip, size=x.shape[-2:], mode="nearest")
                x = x + skip   # 残差相加(不是concat)
        x = self.bottleneck(x)
        x = self.conv_out(x)
        x = self.sigm(x)
        return x


class PathPlannerLoss(nn.Module):
    """路径规划器综合损失 —— 修正穿墙问题的核心。

    三个分量:
      1. L1 路径回归: pred 和 target(路径真值)的 L1 距离
      2. 穿墙惩罚: pred 在障碍格上的激活值
      3. 起终点约束: pred 在起终点处应接近 1
    """

    def __init__(self, wall_weight=5.0, endpoint_weight=2.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.wall_weight = wall_weight
        self.endpoint_weight = endpoint_weight

    def forward(self, pred, target, obs, start_mask, goal_mask):
        l1_loss = self.l1(pred, target)
        wall_loss = (pred * obs).mean()
        ep_loss = ((1.0 - pred) * (start_mask + goal_mask).clamp(0, 1)).mean()
        return l1_loss + self.wall_weight * wall_loss + self.endpoint_weight * ep_loss


class DiceLoss(nn.Module):
    """Dice Loss —— 稀疏细线目标的标配(医学血管分割常用)。

    天然处理类别不平衡: 损失 = 1 - 2|pred∩target| / (|pred|+|target|)
    分母随 pred 总量自适应, 不需要 pos_weight 超参。
    路径占 0.76% 时, Dice 不会被背景主导(和 L1/MSE 不同)。

    可选 smooth 防止 target 全 0 时除零。
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """pred, target: (N, 1, H, W)。pred 是 Sigmoid 后的 [0,1]。"""
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        inter = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()


if __name__ == "__main__":
    m = PPCNet(side=256, n_layers=3, base=64)
    obs = torch.randn(2, 1, 256, 256)
    start = torch.tensor([[100, 50], [200, 200]], dtype=torch.float32)
    goal = torch.tensor([[50, 200], [50, 50]], dtype=torch.float32)
    out = m(obs, start, goal)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"输出: {out.shape}, 参数: {n_params:,} (~{n_params*4/1024/1024:.1f} MB)")
    print(f"输出范围: [{out.min():.3f}, {out.max():.3f}] (Sigmoid后)")
