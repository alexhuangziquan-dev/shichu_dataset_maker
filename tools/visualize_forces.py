# -*- coding: utf-8 -*-
# Batch visualizer for outputs:
#   forces_3ch.npy  (H,W,3) channels [Qx,Qy,R]
#   q.npy           (H,W)
import os, argparse, numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

"""批量可视化脚本（Google 风格文档）。

支持两类输入：
- (H, W, 3) 的 `forces_3ch.npy`：绘制 Qx/Qy/R 热力图、(Qx,Qy) 矢量场、R 的 3D 曲面；
  可选 `--with-mag` 生成 |Q| 的热力图与 3D 曲面。
- (H, W) 的 `q.npy`：绘制 q 的热力图与 3D 曲面。

可对单文件或目录批量渲染；批量模式下可用 `--keep-rel` 保持相对层级到输出目录。
"""

plt.rcParams["figure.dpi"] = 120


def ts_run_root(out_root):
    """生成可视化输出的时间戳目录名（viz_YYYY-MM-DD_HH-MM-SS）。

    Args:
        out_root (str): 可视化输出根目录。

    Returns:
        str: 拼接后的时间戳子目录路径。
    """
    return os.path.join(out_root, datetime.now().strftime("viz_%Y-%m-%d_%H-%M-%S"))


def savefig(fig, outdir, name):
    """安全保存 Matplotlib 图像到 PNG。

    Args:
        fig (matplotlib.figure.Figure): 图对象。
        outdir (str): 目标输出目录。
        name (str): 文件基名（不含扩展名）。

    Notes:
        - 自动创建目标目录。
        - 使用 `bbox_inches="tight"` 以尽量裁剪空白。
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name + ".png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print("[Saved]", path)
    plt.close(fig)


def plot_heatmap(arr, title, cmap="viridis"):
    """绘制二维热力图（origin=lower）。

    Args:
        arr (np.ndarray): 2D 数组。
        title (str): 标题。
        cmap (str, optional): 色图名称。默认为 "viridis"。

    Returns:
        matplotlib.figure.Figure: 图对象。
    """
    fig, ax = plt.subplots()
    im = ax.imshow(arr, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def plot_quiver(Qx, Qy, step=None, scale=None):
    """绘制 (Qx, Qy) 的箭矢图（自动下采样）。

    Args:
        Qx (np.ndarray): 剪力 x 分量 (H,W)。
        Qy (np.ndarray): 剪力 y 分量 (H,W)。
        step (int, optional): 下采样步长；None 时自动估计（~32 箭头/边）。
        scale (float, optional): quiver 的 scale 参数。

    Returns:
        matplotlib.figure.Figure: 图对象。
    """
    H, W = Qx.shape
    if step is None:
        step = max(1, min(H, W) // 32)  # auto downsample
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    U = Qx[::step, ::step]
    V = Qy[::step, ::step]
    fig, ax = plt.subplots()
    ax.quiver(xx, yy, U, V, scale=scale)
    ax.set_title("Force field (Qx,Qy) quiver (step=%d)" % step)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_aspect("equal")
    return fig


def plot_surface3d(arr, title):
    """绘制 3D 曲面图。

    Args:
        arr (np.ndarray): 2D 数组。
        title (str): 标题。

    Returns:
        matplotlib.figure.Figure: 图对象。
    """
    H, W = arr.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, arr, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.set_zlabel("value")
    fig.colorbar(surf, shrink=0.6, aspect=10)
    return fig


def list_target_files(path, recursive, pattern):
    """列举目标路径中的 .npy 文件（可递归、按模式过滤）。

    Args:
        path (str): 文件或目录路径。
        recursive (bool): 是否递归目录。
        pattern (str): 文件名匹配模式（支持 `*.xxx` 或通配 `*`）。

    Returns:
        list[str]: 匹配文件的绝对路径列表。
    """
    targets = []
    if os.path.isfile(path):
        targets.append(path)
    elif os.path.isdir(path):
        if recursive:
            for dirpath, _, files in os.walk(path):
                for fn in files:
                    if fn.lower().endswith(".npy") and _match(fn, pattern):
                        targets.append(os.path.join(dirpath, fn))
        else:
            for fn in os.listdir(path):
                p = os.path.join(path, fn)
                if os.path.isfile(p) and fn.lower().endswith(".npy") and _match(fn, pattern):
                    targets.append(p)
    return targets


def _match(filename, pattern):
    """简单文件名匹配：支持 `*.xxx` 或 `*`；否则使用子串包含。

    Args:
        filename (str): 文件名。
        pattern (str): 模式。

    Returns:
        bool: 是否匹配。
    """
    if pattern in (None, "", "*.npy", "*"):
        return True
    if pattern.startswith("*."):
        return filename.endswith(pattern[1:])
    return pattern in filename


def visualize_forces(npy_path, outdir, with_mag, quiver_step, quiver_scale):
    """可视化 (H,W,3) 的 forces_3ch.npy：Qx/Qy/R/|Q|（可选）与矢量场。

    Args:
        npy_path (str): forces_3ch.npy 的路径。
        outdir (str): 输出目录。
        with_mag (bool): 是否绘制 |Q|。
        quiver_step (int | None): 箭矢下采样步长。
        quiver_scale (float | None): 箭矢 scale。

    Returns:
        bool: 若文件形状不为 (H,W,3) 则返回 False；否则完成绘图并返回 True。
    """
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return False
    Qx, Qy, R = arr[...,0], arr[...,1], arr[...,2]
    tag = os.path.splitext(os.path.basename(npy_path))[0]

    savefig(plot_heatmap(Qx, "Qx (shear x)"), outdir, tag + "_Qx_heatmap")
    savefig(plot_heatmap(Qy, "Qy (shear y)"), outdir, tag + "_Qy_heatmap")
    savefig(plot_heatmap(R,  "R (foundation reaction)"), outdir, tag + "_R_heatmap")

    if with_mag:
        Qmag = np.hypot(Qx, Qy)
        savefig(plot_heatmap(Qmag, "|Q| = sqrt(Qx^2 + Qy^2)"), outdir, tag + "_Qmag_heatmap")

    savefig(plot_quiver(Qx, Qy, step=quiver_step, scale=quiver_scale),
            outdir, tag + "_Q_quiver")

    savefig(plot_surface3d(R, "R 3D surface"), outdir, tag + "_R_surface3d")
    if with_mag:
        Qmag = np.hypot(Qx, Qy)
        savefig(plot_surface3d(Qmag, "|Q| 3D surface"), outdir, tag + "_Qmag_surface3d")
    return True


def visualize_q(npy_path, outdir):
    """可视化 (H,W) 的 q.npy：热力图与 3D 曲面。

    Args:
        npy_path (str): q.npy 路径。
        outdir (str): 输出目录。

    Returns:
        bool: 若文件形状不为 (H,W) 则返回 False；否则完成绘图并返回 True。
    """
    arr = np.load(npy_path)
    if arr.ndim != 2:
        return False
    tag = os.path.splitext(os.path.basename(npy_path))[0]
    savefig(plot_heatmap(arr, "q (external load per unit area)"), outdir, tag + "_q_heatmap")
    savefig(plot_surface3d(arr, "q 3D surface"), outdir, tag + "_q_surface3d")
    return True


def main():
    """命令行入口：批量/单文件可视化 forces 与 q。"""
    ap = argparse.ArgumentParser(description="Batch visualize outputs: forces_3ch.npy and q.npy.")
    ap.add_argument("--input", required=True, help="a .npy file OR a directory")
    ap.add_argument("--outroot", default="./viz_out", help="root folder for visualization outputs (timestamped)")
    ap.add_argument("--pattern", default="*.npy", help="file name pattern to match when input is a directory")
    ap.add_argument("--recursive", action="store_true", help="recurse into subfolders")
    ap.add_argument("--keep-rel", dest="keep_rel", action="store_true",
                    help="preserve relative subfolder structure under outroot")
    ap.add_argument("--with-mag", action="store_true",
                    help="also draw |Q| heatmap and surface for forces")
    ap.add_argument("--quiver-step", type=int, default=None, help="downsample step for quiver")
    ap.add_argument("--quiver-scale", type=float, default=None, help="scale for quiver")
    args = ap.parse_args()

    run_root = ts_run_root(args.outroot)
    os.makedirs(run_root, exist_ok=True)

    targets = list_target_files(args.input, args.recursive, args.pattern)
    print("[Info] Found", len(targets), "file(s)")
    base_dir = os.path.abspath(args.input) if os.path.isdir(args.input) else None

    for p in targets:
        try:
            # Windows 友好：保持输入目录的相对层级到输出
            rel_dir = ""
            if args.keep_rel and base_dir:
                try:
                    rel_dir = os.path.dirname(os.path.relpath(os.path.abspath(p), base_dir))
                    if rel_dir.startswith(".."):
                        rel_dir = ""
                except Exception:
                    rel_dir = ""
            outdir = os.path.join(run_root, rel_dir)
            os.makedirs(outdir, exist_ok=True)

            # 优先按 forces 解析，失败再尝试 q
            ok = visualize_forces(p, outdir, args.with_mag, args.quiver_step, args.quiver_scale)
            if not ok:
                visualize_q(p, outdir)
        except Exception as e:
            print("[Error]", p, e)

    print("[DONE] Visualizations saved under:", run_root)


if __name__ == "__main__":
    main()
