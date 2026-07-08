"""
export_ppcnet_onnx.py — 把势场 PPCNet (ep40) 导出为 ONNX, 用于 hb_mapper checker。

关键挑战: PPCNet.forward(obs, start, goal) 接受坐标输入, 这是动态的。
BPU 要求静态输入 shape。解决方案: 把 GaussianRelativePE 在导出前预计算成 channel,
让模型只接受 (N,3,H,W) 图像输入 [障碍 + pe_start + pe_goal], 坐标在外部算好PE再传入。

这样 BPU 只跑纯卷积(Conv/BN/ReLU/ConvTranspose/Add), 坐标→PE的转换在CPU做。
也正好对应知识库的 encoder-only 兜底思路: PE生成+decoder可能在CPU, BPU跑核心卷积。

导出两个版本:
  v1 完整版: 整个 PPCNet (含 ConvTranspose decoder) → 测 ConvTranspose 是否支持
  v2 encoder-only: 只 encoder (纯 Conv/BN/ReLU/Pool) → 兜底, 必过
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from ppcnet_model import PPCNet, GaussianRelativePE


class PPCNetStaticInput(nn.Module):
    """包装 PPCNet: 接受预计算的 PE 通道, 输出势场。
    输入: (N, 3, H, W) = [障碍图, pe_start, pe_goal]
    输出: (N, 1, H, W) 势场
    这样把坐标→PE的计算移到 CPU, BPU 只跑卷积。"""

    def __init__(self, base=64, n_layers=3):
        super().__init__()
        self.sigm = nn.Identity()  # 势场回归, 不用 sigmoid
        n_channels = [base * (2 ** i) for i in range(n_layers)]

        class _ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch, transpose=False, last_output_pad=1):
                super().__init__()
                self.activation = nn.ReLU()
                if transpose:
                    self.bn1 = nn.BatchNorm2d(out_ch)
                    self.bn2 = nn.BatchNorm2d(out_ch)
                    self.bn3 = nn.BatchNorm2d(out_ch // 2)
                    self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                    self.conv2 = nn.ConvTranspose2d(out_ch, out_ch, 3, padding=1)
                    self.conv3 = nn.ConvTranspose2d(out_ch, out_ch // 2, 3,
                                                    stride=2, padding=1,
                                                    output_padding=last_output_pad)
                else:
                    self.bn1 = nn.BatchNorm2d(out_ch)
                    self.bn2 = nn.BatchNorm2d(out_ch)
                    self.bn3 = nn.BatchNorm2d(out_ch * 2)
                    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
                    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
                    self.conv3 = nn.Conv2d(out_ch, out_ch * 2, 3, stride=2, padding=1)
                self._fw = nn.Sequential(
                    self.conv1, self.activation, self.bn1,
                    self.conv2, self.activation, self.bn2,
                    self.conv3, self.activation, self.bn3)

            def forward(self, x):
                return self._fw(x)

        self.conv_down = nn.ModuleList([
            _ConvBlock(c if i > 0 else 3, c) for i, c in enumerate(n_channels)])
        self.conv_up = nn.ModuleList([
            _ConvBlock(2 * c, 2 * c, transpose=True, last_output_pad=1)
            for i, c in enumerate(n_channels[::-1])])
        self.bottleneck = nn.Conv2d(base, 1, 3, padding=1)
        self.conv_out = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        """x: (N,3,H,W) [障碍, pe_start, pe_goal]"""
        skip_conn = []
        for i, conv in enumerate(self.conv_down):
            x = conv(x)
            if i < len(self.conv_down) - 1:
                skip_conn.append(x)
        for i, conv in enumerate(self.conv_up):
            x = conv(x)
            if i < len(skip_conn):
                import torch.nn.functional as Fn
                skip = skip_conn[-1 - i]
                if x.shape[-1] != skip.shape[-1]:
                    skip = Fn.interpolate(skip, size=x.shape[-2:], mode="nearest")
                x = x + skip
        x = self.bottleneck(x)
        x = self.conv_out(x)
        return self.sigm(x)


class PPCNetEncoderOnly(nn.Module):
    """兜底版: 只保留 encoder + bottleneck, 用 stride 卷积下采样后直接 1x1 卷积输出。
    无 ConvTranspose, 无 F.interpolate。全是 BPU 必支持的算子。
    输出分辨率会比输入小(下采样), CPU 端 F.interpolate 上采样回去。
    目的: 验证 BPU 算子兼容性的下界(如果这个都过不了, 整个方案要重设计)。"""

    def __init__(self, base=64, n_layers=3):
        super().__init__()
        n_channels = [base * (2 ** i) for i in range(n_layers)]

        class _ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch * 2, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch * 2), nn.ReLU())

            def forward(self, x):
                return self.net(x)

        self.encoder = nn.ModuleList([
            _ConvBlock(c if i > 0 else 3, c) for i, c in enumerate(n_channels)])
        # 最后用一个 1x1 conv 把通道压到 1 (在最低分辨率上输出势场粗略值)
        self.head = nn.Conv2d(n_channels[-1] * 2, 1, 1)

    def forward(self, x):
        for conv in self.encoder:
            x = conv(x)
        return self.head(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_hybrid/ep40.pt")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    base = ck.get("base", args.base)

    # === v1: 完整版 (含 ConvTranspose) ===
    print("[export] 导出 v1 完整版 (含 ConvTranspose decoder)...")
    full = PPCNetStaticInput(base=base, n_layers=3)
    # 从原 ckpt 加载权重 (key 名基本一致, 因为结构相同)
    missing, unexpected = full.load_state_dict(ck["model"], strict=False)
    # 过滤掉原模型的 pe/sigm (我们重定义了)
    real_missing = [k for k in missing if not k.startswith(("pe.", "sigm"))]
    if real_missing:
        print(f"[export] ⚠️ 完整版缺失 key: {real_missing[:5]}...")
    full.eval()
    dummy_full = torch.randn(1, 3, 256, 256)
    out_full = os.path.join(args.out_dir, "ppcnet_full.onnx")
    torch.onnx.export(full, dummy_full, out_full,
                      input_names=["input"], output_names=["field"],
                      opset_version=args.opset, do_constant_folding=True,
                      dynamo=False)
    print(f"[export] v1: {out_full}  输入(1,3,256,256) 输出(1,1,256,256)")

    # === v2: encoder-only (兜底) ===
    print("[export] 导出 v2 encoder-only (兜底, 纯Conv/BN/ReLU)...")
    enc = PPCNetEncoderOnly(base=base, n_layers=3)
    enc.eval()
    dummy_enc = torch.randn(1, 3, 256, 256)
    out_enc = os.path.join(args.out_dir, "ppcnet_encoder.onnx")
    torch.onnx.export(enc, dummy_enc, out_enc,
                      input_names=["input"], output_names=["field_coarse"],
                      opset_version=args.opset, do_constant_folding=True,
                      dynamo=False)
    print(f"[export] v2: {out_enc}  输入(1,3,256,256) 输出(1,1,32,32)")

    # === 强制降低 ONNX IR 版本(地平线工具链要 IR<=9) ===
    # torch 2.9 默认生成 IR=10, hb_compile 4.7.5 只认 <=9
    import onnx
    for path in [out_full, out_enc]:
        m = onnx.load(path)
        m.ir_version = 9
        onnx.save(m, path)
    print("[export] 已强制 IR 版本 → 9 (兼容 hb_compile 4.7.5)")

    # === sanity check ===
    try:
        import onnxruntime as ort
        for name, path, dummy in [("v1完整", out_full, dummy_full),
                                   ("v2encoder", out_enc, dummy_enc)]:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"input": dummy.numpy()})[0]
            print(f"[export] {name} ONNX sanity: 输出shape={ort_out.shape} ✅")
    except ImportError:
        print("[export] 无 onnxruntime, 跳过 sanity")

    print(f"\n[export] 完成。下一步在容器里:")
    print(f"  hb_compile -m {out_full} --march nash-m -i input 1x3x256x256 --skip compile")


if __name__ == "__main__":
    main()
