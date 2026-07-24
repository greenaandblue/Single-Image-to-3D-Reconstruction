# Single-Image-to-3D-Reconstruction

单目视觉 3D 重建系统：从一张 RGB 图片估计深度，重建出点云 / 三角网格模型。

```
RGB Image
    |
    v
Depth Estimation (Depth Anything V2 / MiDaS)
    |
    v
Depth Map
    |
    v
Point Cloud  (针孔相机模型反投影)
    |
    v
Triangle Mesh (深度图直接三角化)
    |
    v
.ply / .obj  ->  Blender / MeshLab / Open3D / Unity
```

## 目录结构

```
img23d/
├── demo.py                  # 主入口
├── requirements.txt
├── src/
│   ├── depth_estimation.py  # 深度估计 (Depth Anything V2 / MiDaS)
│   ├── geometry.py          # 深度图 -> 点云 / 网格 (纯 numpy 实现)
│   ├── io_utils.py          # 读图 / 存 obj / ply
│   └── visualize.py         # matplotlib 预览 + open3d 交互查看
└── examples/
    └── test_input.jpg       # 测试图片
```

## 安装

```bash
pip install -r requirements.txt
```

第一次跑 `depth_anything` 或 `midas` 会自动从 HuggingFace / torch hub 下载模型权重，需要联网。

## 使用

```bash
# 完整流程：图片 -> 深度 -> 点云 + 网格
python demo.py --image examples/test_input.jpg --out outputs --backend depth_anything

# 只要点云
python demo.py --image your_photo.jpg --out outputs --mode pointcloud

# 只要网格 (.obj，可以直接拖进 Blender)
python demo.py --image your_photo.jpg --out outputs --mode mesh

# 没装 torch，想先测试点云/网格代码本身对不对：
python demo.py --image examples/test_input.jpg --out outputs --use_fake_depth
```

输出：

- `outputs/depth.png` — 深度图（灰度，越亮越远）
- `outputs/pointcloud.ply` — 点云，可用 MeshLab / Open3D / Blender 打开
- `outputs/mesh.obj` — 三角网格，可直接拖进 Blender / Unity / Godot
- `outputs/pointcloud_preview.png` — 静态预览图（不需要任何 3D 软件就能看效果）

## 原理说明

### 1. 深度估计

`src/depth_estimation.py` 里的 `DepthEstimator` 支持两个后端：

- `depth_anything`（推荐）：HuggingFace 上的 `Depth-Anything-V2-Small`，速度快、效果好，CPU 也能跑（会比较慢）。
- `midas`：`intel-isl/MiDaS` 的 `MiDaS_small`，通过 `torch.hub` 加载。

两者输出的都是**相对深度**（不是米/厘米这种绝对尺度）——单张 2D 图片本质上无法恢复真实世界的绝对尺度，这是单目重建的固有限制。

### 2. 深度图 → 点云

`src/geometry.py` 里的 `depth_to_pointcloud`，用针孔相机模型：

```
Z = depth
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

因为没有真实相机的内参（fx, fy, cx, cy），代码里用经验值估算（`CameraIntrinsics` 默认 `fx = fy = width * 1.2`），这样重建出来的形状比例是合理的，但不代表真实物理尺寸。如果你知道拍照手机的真实焦距/传感器参数，可以在 `CameraIntrinsics(...)` 里手动传入更准确的值。

### 3. 点云 → 网格

没有用 Poisson 重建（那个更适合处理稠密、多视角扫描得到的点云），而是直接把深度图按像素网格三角化（每个 2×2 像素格子拆成 2 个三角形），这是单张深度图转 mesh 最直接、最快的方法。为了避免物体边缘和背景被错误地"焊"在一起形成拉长的穿帮面片，加入了深度断层检测（`edge_threshold_ratio`），深度跳变过大的三角形会被跳过。

## 局限性（诚实说明）

- 单张图片**无法**恢复背面/被遮挡部分的几何 —— 输出的是一个"深度浮雕"式的 2.5D 网格，而不是完整的封闭 3D 模型。真正的完整 3D 需要多视角（参考下面"进阶方向"）。
- 深度是相对值，没有真实尺度。
- 相机内参是估算值，不是真实标定结果。

## 进阶方向（如果想做成更完整的研究/作品集项目）

1. **实时版本**：把 `DepthEstimator` 换成摄像头逐帧输入，做成实时深度可视化 demo（适合展示 Robotics/CV 背景）。
2. **多视角融合**：拍摄同一物体的多张照片，各自反投影出点云后用 ICP 对齐、合并，能重建出更完整的模型。
3. **NeRF / Instant-NGP**：用 20-50 张多视角照片训练一个神经辐射场，是目前更前沿的做法，能得到照片级真实的完整 3D 场景，但训练成本更高。
4. **Open3D 后处理**：如果本地装了 `open3d`，可以用它做点云降噪（statistical outlier removal）、法线估计 + Poisson 重建，得到更平滑封闭的网格。
