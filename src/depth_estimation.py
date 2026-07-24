"""
depth_estimation.py
--------------------
单目深度估计模块。

输入: 一张 RGB 图像 (H, W, 3)
输出: 一张与原图同尺寸的深度图 (H, W)，float32，数值越大代表越"远"（相对深度）

支持两种后端：
  1. "depth_anything"  -> 使用 HuggingFace transformers 的 Depth Anything V2 (推荐, 效果最好)
  2. "midas"           -> 使用 torch.hub 加载 MiDaS (备用方案)

注意：
  这两种后端都需要 torch + 联网下载预训练权重（第一次运行时）。
  如果你在没有 GPU / 没有网络的环境测试整条 pipeline，
  可以用 examples/generate_test_scene.py 生成一张"假"的深度图，
  跳过这一步，直接验证后面的点云/网格重建代码是否正确。
"""

from __future__ import annotations
from typing import Any, cast
from torch import nn
import numpy as np


class DepthEstimator:
    def __init__(self, backend: str = "depth_anything", device: str | None = None):
        """
        backend: "depth_anything" 或 "midas"
        device:  "cuda" / "cpu" / "mps"（Apple Silicon）。默认自动选择。
        """
        self.backend = backend
        self._model: nn.Module | None = None
        self._processor: Any = None

        import torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self._torch = torch

        if backend == "depth_anything":
            self._load_depth_anything()
        elif backend == "midas":
            self._load_midas()
        else:
            raise ValueError(f"未知 backend: {backend!r}，请用 'depth_anything' 或 'midas'")

    # ------------------------------------------------------------------
    def _load_depth_anything(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)

        assert isinstance(model, nn.Module)

        self._model = model.to(self.device).eval()

    def _load_midas(self):
        torch = self._torch

        model: Any = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small"
        )

        self._model = model.to(self.device).eval()

    # ------------------------------------------------------------------
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        image: (H, W, 3) uint8 RGB
        return: (H, W) float32 相对深度图（数值越大越远）
        """
        torch = self._torch
        h, w = image.shape[:2]
        

        if self.backend == "depth_anything":
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            if self._model is None:
                raise RuntimeError("Depth model is not loaded")

            with torch.no_grad():
                outputs: Any = self._model(**inputs)
                predicted_depth = outputs.predicted_depth  # (1, h', w')
            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
            # Depth Anything 输出的是 "disparity-like" 值：越大越近，这里转成越大越远，方便统一约定
            depth = depth.max() - depth
        else:  # midas
            input_batch = self._processor(image).to(self.device)
            if self._model is None:
                raise RuntimeError("Depth model is not loaded")

            with torch.no_grad():
                prediction = self._model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze()
            depth = prediction.cpu().numpy()
            depth = depth.max() - depth  # MiDaS 同理，输出越大越近，这里翻转

        return depth.astype(np.float32)
