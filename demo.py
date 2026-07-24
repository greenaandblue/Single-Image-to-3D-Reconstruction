"""
demo.py
--------
完整 pipeline 入口：

    RGB 图片
      -> 深度估计 (Depth Anything V2 / MiDaS)
      -> 点云 (point cloud)
      -> 三角网格 (mesh)
      -> 导出 .ply / .obj
      -> 生成预览图 preview.png

用法:
    python demo.py --image path/to/photo.jpg --out outputs/ --backend depth_anything

如果你还没装 torch/transformers，想先测试整条“点云/网格”代码是否正确，
可以用 --use_fake_depth 参数，跳过深度估计模型，用一张合成的深度图代替：

    python demo.py --image path/to/photo.jpg --out outputs/ --use_fake_depth
"""

from __future__ import annotations
import argparse
import os
import numpy as np

from src.geometry import CameraIntrinsics, depth_to_pointcloud, depth_to_mesh
from src.io_utils import load_image, save_depth_png, save_point_cloud_ply, save_mesh_obj
from src.visualize import preview_matplotlib


def make_fake_depth(image: np.ndarray) -> np.ndarray:
    """
    仅用于离线测试：生成一个"碗状"合成深度图（中心近、四周远），
    不代表任何真实场景，只是为了验证点云/网格生成代码本身没问题。
    """
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    depth = 1.0 + r * 2.0  # 中心 depth=1（近），边缘更大（远）
    return depth.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument("--out", default="outputs", help="输出目录")
    ap.add_argument("--backend", default="depth_anything", choices=["depth_anything", "midas"])
    ap.add_argument("--use_fake_depth", action="store_true",
                     help="跳过深度模型，用合成深度图测试pipeline（不需要torch）")
    ap.add_argument("--mode", default="both", choices=["pointcloud", "mesh", "both"])
    ap.add_argument("--max_points", type=int, default=200000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/5] 读取图片: {args.image}")
    image = load_image(args.image)
    h, w = image.shape[:2]
    print(f"      尺寸: {w}x{h}")

    print(f"[2/5] 深度估计 (backend={'FAKE(测试用)' if args.use_fake_depth else args.backend})")
    if args.use_fake_depth:
        depth = make_fake_depth(image)
    else:
        from src.depth_estimation import DepthEstimator
        estimator = DepthEstimator(backend=args.backend)
        depth = estimator.estimate(image)

    depth_png_path = os.path.join(args.out, "depth.png")
    save_depth_png(depth, depth_png_path)
    print(f"      深度图已保存: {depth_png_path}")

    intr = CameraIntrinsics(width=w, height=h)
    print(f"[3/5] 相机内参 (估算): {intr}")

    if args.mode in ("pointcloud", "both"):
        print("[4/5] 生成点云...")
        points, colors = depth_to_pointcloud(depth, image, intr, max_points=args.max_points)
        ply_path = os.path.join(args.out, "pointcloud.ply")
        save_point_cloud_ply(points, colors, ply_path)
        print(f"      点云已保存: {ply_path}  (点数: {points.shape[0]})")

        preview_path = os.path.join(args.out, "pointcloud_preview.png")
        preview_matplotlib(points, colors, preview_path)
        print(f"      预览图已保存: {preview_path}")

    if args.mode in ("mesh", "both"):
        print("[5/5] 生成三角网格...")
        vertices, faces, vcolors = depth_to_mesh(depth, image, intr)
        obj_path = os.path.join(args.out, "mesh.obj")
        save_mesh_obj(vertices, faces, vcolors, obj_path)
        print(f"      网格已保存: {obj_path}  (顶点: {vertices.shape[0]}, 面片: {faces.shape[0]})")

    print("\n完成！可以用 Blender / MeshLab / Open3D 打开 outputs/ 里的 .obj 或 .ply 文件查看3D模型。")


if __name__ == "__main__":
    main()
