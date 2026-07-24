"""
geometry.py
------------
核心几何计算：把 (RGB图像 + 深度图) 变成 3D 点云 / 三角网格。

数学原理（针孔相机模型）：
    对每个像素 (u, v)，其深度为 Z：
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    (fx, fy) 是焦距（像素单位），(cx, cy) 是主点（一般是图像中心）。

    如果没有真实相机内参，可以用一个经验值估算：
        fx = fy = w （或 w * 1.2 左右），cx = w/2, cy = h/2
    这不是真实尺度，只是为了让重建出来的形状看起来合理（unit depth）。

本模块完全用 numpy 实现，不依赖 Open3D，方便在任何环境下测试。
"""

from __future__ import annotations
import numpy as np


class CameraIntrinsics:
    def __init__(self, width: int, height: int, fx: float | None = None,
                 fy: float | None = None, cx: float | None = None, cy: float | None = None):
        self.width = width
        self.height = height
        self.fx = fx if fx is not None else width * 1.2
        self.fy = fy if fy is not None else self.fx
        self.cx = cx if cx is not None else width / 2.0
        self.cy = cy if cy is not None else height / 2.0

    def __repr__(self):
        return f"CameraIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f})"


def depth_to_pointcloud(depth: np.ndarray, image: np.ndarray | None, intr: CameraIntrinsics,
                         depth_scale: float = 1.0, max_points: int | None = None):
    """
    depth: (H, W) float32
    image: (H, W, 3) uint8 或 None（没有颜色时点云为灰色）
    return: points (N,3) float32, colors (N,3) float32 in [0,1]
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) * depth_scale
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy

    points = np.stack([x, -y, -z], axis=-1).reshape(-1, 3)  # 翻转 y/z 让它符合常见右手坐标系（Y向上）

    if image is not None:
        colors = image.reshape(-1, 3).astype(np.float32) / 255.0
    else:
        colors = np.full((points.shape[0], 3), 0.6, dtype=np.float32)

    if max_points is not None and points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points, colors = points[idx], colors[idx]

    return points, colors


def depth_to_mesh(depth: np.ndarray, image: np.ndarray | None, intr: CameraIntrinsics,
                   depth_scale: float = 1.0, edge_threshold_ratio: float = 0.05):
    """
    把深度图直接三角化成网格（每 2x2 像素格子拆成 2 个三角形）。
    这是单张深度图转 mesh 最直接、最快的方法，不需要 Poisson 重建。

    edge_threshold_ratio: 当一个三角形三个顶点深度差异超过
        (平均深度 * edge_threshold_ratio) 时，认为这是前景/背景的"深度断层"，
        跳过该三角形，避免生成拉长的"穿帮"面片。

    return: vertices (N,3) float32, faces (M,3) int32, vertex_colors (N,3) float32
    """
    h, w = depth.shape
    points, colors = depth_to_pointcloud(depth, image, intr, depth_scale=depth_scale)

    def vid(row, col):
        return row * w + col

    faces = []
    for r in range(h - 1):
        for c in range(w - 1):
            i00, i01 = vid(r, c), vid(r, c + 1)
            i10, i11 = vid(r + 1, c), vid(r + 1, c + 1)

            z00, z01, z10, z11 = depth[r, c], depth[r, c + 1], depth[r + 1, c], depth[r + 1, c + 1]

            # 三角形 1: (i00, i10, i11)
            zs = (z00, z10, z11)
            if max(zs) - min(zs) < np.mean(zs) * edge_threshold_ratio:
                faces.append((i00, i10, i11))

            # 三角形 2: (i00, i11, i01)
            zs = (z00, z11, z01)
            if max(zs) - min(zs) < np.mean(zs) * edge_threshold_ratio:
                faces.append((i00, i11, i01))

    faces = np.array(faces, dtype=np.int32)
    return points.astype(np.float32), faces, colors.astype(np.float32)
