"""
visualize.py
-------------
两种可视化方式：

1. preview_matplotlib(...)  -> 生成静态预览图 (png)，任何环境都能跑，
   适合快速检查重建结果对不对，不需要 Open3D。

2. view_open3d(...)         -> 打开一个可以旋转/缩放的交互式 3D 窗口，
   效果更好，但需要本地安装 open3d 并且有图形界面（不适合纯服务器环境）。
"""

from __future__ import annotations
import numpy as np


def preview_matplotlib(points: np.ndarray, colors: np.ndarray, save_path: str,
                        max_points: int = 20000, elev: int = -70, azim: int = -90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points, colors = points[idx], colors[idx]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1.2, depthshade=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor="black")
    plt.close(fig)


def view_open3d(points: np.ndarray, colors: np.ndarray):
    """需要本地环境: pip install open3d，且要有显示器/图形界面。"""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
