# -*- coding: utf-8 -*-
"""
从位移场 w[m] 反算力学场（稳健双通道版）
====================================

设计要点
--------
- 轻处理通道：用于 Qx、Qy、R，尽量保留中高频细节（减少过度平滑对剪力的削弱）。
- 强处理通道：用于 q，重点抑制双拉普拉斯（∇⁴）对高频噪声的放大效应。
- 两条通道可独立配置：高斯平滑、下采样、频域低通。
- **无论内部如何处理，最终输出尺寸固定为原图 H×W（例如 256×256）。**

力学定义（Pasternak 基床薄板，小挠度 Kirchhoff-Love 假设）
----------------------------------------------------------
令位移场为 w(x,y) [m]，板弯曲刚度
    D = E h^3 / (12 (1 - ν^2))   [N·m]
PDE 平衡方程：
    D ∇⁴ w + G_p ∇² w + k w = q          (1)
其中：
- ∇² 为 2D 拉普拉斯；∇⁴ = ∇²(∇²) 为双拉普拉斯；
- q [N/m²] 为分布载荷；
- k [N/m³] 为 Winkler 弹簧系数；
- G_p [Pa·m = N/m] 为 Pasternak 剪切层参数；
- E [Pa], ν [-], h [m] 分别为弹性模量、泊松比、板厚。

剪力向量（Kirchhoff 剪力）：
    Q = (Qx, Qy) = - D ∇(∇² w)          (2)
反力合力（基床反力）：
    R = k w + G_p ∇² w                  (3)
故分布载荷由(1)可写为：
    q = D ∇⁴ w + G_p ∇² w + k w          (4)

单位约定
--------
- w: m
- Qx, Qy: N/m
- R, q: N/m²

数值实现摘要
------------
- ∇² 使用五点差分（拉普拉斯 5 点 stencil），边界复制外推；
- ∂/∂x, ∂/∂y 使用中心差分，边界一阶前/后向；
- 频域低通在 rFFT 平面上用半径截止 + 余弦窗软过渡；
- 下采样优先使用块平均以抗混叠；上采样使用最近邻保证尺寸一致；
- 强通道（q）中更强的平滑 / 低通 / 下采样降低 ∇⁴ 的噪声放大。

"""

import argparse
import os
import json
from datetime import datetime

import yaml
import numpy as np


# -------------------------
# 基础工具
# -------------------------
def timestamp() -> str:
    """返回带时间戳的可读运行目录名。

    Returns
    -------
    str
        形如 "run_2025-09-13_11-05-42" 的目录名。

    Notes
    -----
    - 提供可追溯的输出目录名，不涉数值计算。
    """
    return datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")


def load_config(path: str) -> dict:
    """加载 YAML 配置文件并返回字典。

    Parameters
    ----------
    path : str
        YAML 文件路径。

    Returns
    -------
    dict
        解析后的配置字典；若文件为空返回 {}。

    Raises
    ------
    FileNotFoundError
        当路径不存在时。
    yaml.YAMLError
        YAML 解析失败时。

    Notes
    -----
    - 仅 I/O 工具函数，不涉及力学公式。
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def iter_w_files(root_dir: str, pattern: str, recursive: bool):
    """遍历目录下满足后缀模式的位移场文件路径。

    Parameters
    ----------
    root_dir : str
        根目录。
    pattern : str
        通配后缀（仅按后缀过滤，如 "*.npy"）。
    recursive : bool
        是否递归子目录。

    Yields
    ------
    str
        满足条件的文件绝对路径。

    Notes
    -----
    - 文件收集工具，不涉及数学计算。
    """
    if not os.path.isdir(root_dir):
        return
    if not pattern:
        pattern = "*.npy"
    suffix = pattern.replace("*", "").lower()
    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(suffix):
                yield os.path.join(dirpath, fn)
        if not recursive:
            break


# -------------------------
# 预处理：平滑 / 下采样 / 上采样 / 低通
# -------------------------
def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    """构建并返回归一化的一维高斯核。

    Parameters
    ----------
    sigma : float
        标准差像素单位；σ<=0 时退化为 [1].

    Returns
    -------
    np.ndarray
        归一化核，长度约 6σ+1，半径 round(3σ)。

    数学公式
    --------
    核定义：
        k(x) = exp(-x^2 / (2 σ^2))，再做 L1 归一化。

    Notes
    -----
    - 用于可分离高斯卷积的 1D 核。
    """
    if sigma is None or sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(round(3.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def gaussian_blur(w: np.ndarray, sigma: float) -> np.ndarray:
    """对二维场应用可分离高斯模糊。

    Parameters
    ----------
    w : np.ndarray
        输入二维场（此处为位移 w）。
    sigma : float
        高斯标准差（像素）；σ<=0 时原样返回。

    Returns
    -------
    np.ndarray
        模糊后的场，保持 dtype。

    数学公式
    --------
    可分离卷积：
        w_g = (w * k_x) * k_y
    其中 k_x = k_y 为 `_gaussian_kernel1d(σ)`。

    边界条件
    --------
    - 使用反射填充（'reflect'）以减小边缘伪影。

    复杂度
    ------
    - O(H W · K)，K≈6σ+1。较 2D 卷积更高效。

    目的
    ----
    - 抑制高频噪声，特别是在计算 ∇² 和 ∇⁴ 前。
    """
    if sigma is None or sigma <= 0:
        return w
    k = _gaussian_kernel1d(sigma)
    pad = len(k) // 2
    # x 方向卷积
    wpad = np.pad(w, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.empty_like(wpad, dtype=np.float64)
    for i in range(wpad.shape[0]):
        tmp[i, :] = np.convolve(wpad[i, :], k, mode="same")
    tmp = tmp[:, pad:-pad]
    # y 方向卷积
    wpad = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(wpad, dtype=np.float64)
    for j in range(wpad.shape[1]):
        out[:, j] = np.convolve(wpad[:, j], k, mode="same")
    out = out[pad:-pad, :]
    return out.astype(w.dtype, copy=False)


def downsample(w: np.ndarray, stride: int, mode: str = "mean") -> np.ndarray:
    """以块平均（抗混叠）或抽取方式对二维场下采样。

    Parameters
    ----------
    w : np.ndarray
        输入场。
    stride : int
        步长；<=1 时原样返回。
    mode : {"mean","decimate"}
        - "mean": s×s 块平均，抗混叠；
        - "decimate": 直接抽取。

    Returns
    -------
    np.ndarray
        下采样场。

    Notes
    -----
    - q 通道常用更大的 stride 以抑制 ∇⁴ 的放大噪声。
    """
    if stride is None or stride <= 1:
        return w
    s = int(stride)
    H, W = w.shape
    Hs = (H // s) * s
    Ws = (W // s) * s
    w = w[:Hs, :Ws]
    if mode == "decimate":
        return w[::s, ::s]
    # 抗混叠的块平均
    return w.reshape(Hs // s, s, Ws // s, s).mean(axis=(1, 3))


def upsample_nearest(w_small: np.ndarray, target_hw) -> np.ndarray:
    """使用最近邻插值将二维场上采样到目标尺寸。

    Parameters
    ----------
    w_small : np.ndarray
        小尺寸场。
    target_hw : Tuple[int,int]
        目标 (H, W)。

    Returns
    -------
    np.ndarray
        上采样后裁剪到恰好 (H,W)。

    设计动机
    --------
    - 使用最近邻是为**严格保持数值幅度**（不引入插值平滑），
      便于与轻通道/强通道在物理量上的一致比较。
    """
    Ht, Wt = target_hw
    hs, ws = w_small.shape
    ry = int(np.ceil(Ht / hs))
    rx = int(np.ceil(Wt / ws))
    big = np.repeat(np.repeat(w_small, ry, axis=0), rx, axis=1)
    return big[:Ht, :Wt]


def fft_lowpass2d(w: np.ndarray, ratio: float = 0.15, soft: bool = True) -> np.ndarray:
    """在频域对二维场施加各向同性低通滤波。

    Parameters
    ----------
    w : np.ndarray
        输入二维场。
    ratio : float, optional
        截止频率相对 Nyquist 的比例，0<ratio<0.5；默认 0.15。
    soft : bool, optional
        若为 True，使用余弦窗在 [0.9*cut, cut] 软过渡。

    Returns
    -------
    np.ndarray
        低通后的场，保持 dtype。

    数学说明
    --------
    - 实数 FFT: `rfft2` 把频谱限制在 k_x∈[0, 0.5]。
    - 构建半径谱 K = sqrt(Kx^2 + Ky^2)，设置半径截止：
        mask(K) = 1 (K<=k0), 0.5(1+cos(π (K-k0)/(k_c-k0))) (k0<=K<=k_c), 0 (K>k_c)
      其中 k0 = 0.9 k_c（若 soft=True），k_c = ratio。

    目的
    ----
    - 进一步压制高频（尤其用于 q 通道以稳定 ∇⁴）。
    """
    if ratio is None or ratio <= 0 or ratio >= 0.5:
        return w
    H, W = w.shape
    Wf = np.fft.rfft2(w)  # 实数优化
    ky = np.fft.fftfreq(H)          # [-0.5, 0.5)
    kx = np.fft.rfftfreq(W)         # [0, 0.5]
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)
    k_cut = float(ratio)
    if soft:
        k0 = 0.9 * k_cut
        mask = np.ones_like(K)
        mask[K > k_cut] = 0.0
        trans = (K >= k0) & (K <= k_cut)
        mask[trans] = 0.5 * (1 + np.cos(np.pi * (K[trans] - k0) / (k_cut - k0)))
    else:
        mask = (K <= k_cut).astype(Wf.real.dtype)
    Wf *= mask
    w_lp = np.fft.irfft2(Wf, s=w.shape)
    return w_lp.astype(w.dtype, copy=False)


# -------------------------
# 有限差分算子（明确、稳健）
# -------------------------
def laplacian_5pt(w: np.ndarray, dx: float) -> np.ndarray:
    """用五点差分在二维网格上计算拉普拉斯算子。

    Parameters
    ----------
    w : np.ndarray
        标量场（此处为位移 w）。
    dx : float
        网格间距 [m]（假设 x=y=a/(N-1) 等距）。

    Returns
    -------
    np.ndarray
        ∇² w 的离散估计。

    数学公式
    --------
    2D 五点差分：
        ∇² w(i,j) ≈ [w(i+1,j)+w(i-1,j)+w(i,j+1)+w(i,j-1)-4 w(i,j)] / dx²

    边界条件
    --------
    - 复制边界外推（Neumann 近似），抑制边界振铃。

    误差
    ----
    - 截断误差 O(dx²)；噪声会被二阶导数放大，需配合平滑/低通。
    """
    up = np.vstack([w[0:1, :], w[:-1, :]])
    down = np.vstack([w[1:, :], w[-1:, :]])
    left = np.hstack([w[:, 0:1], w[:, :-1]])
    right = np.hstack([w[:, 1:], w[:, -1:]])
    return (up + down + left + right - 4.0 * w) / (dx * dx)


def grad_central_x(w: np.ndarray, dx: float) -> np.ndarray:
    """用中心差分（边界一阶回退）计算 x 方向一阶导数。

    Parameters
    ----------
    w : np.ndarray
        标量场。
    dx : float
        网格间距 [m]。

    Returns
    -------
    np.ndarray
        ∂w/∂x 的离散估计。

    数学公式
    --------
    - 内点： (w(i,j+1) - w(i,j-1)) / (2 dx)
    - 左边界： (w(i,1) - w(i,0)) / dx
    - 右边界： (w(i,-1) - w(i,-2)) / dx

    精度
    ----
    - 内点二阶，边界一阶。
    """
    gx = np.empty_like(w, dtype=np.float64)
    gx[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2.0 * dx)
    gx[:, 0] = (w[:, 1] - w[:, 0]) / dx
    gx[:, -1] = (w[:, -1] - w[:, -2]) / dx
    return gx


def grad_central_y(w: np.ndarray, dy: float) -> np.ndarray:
    """用中心差分（边界一阶回退）计算 y 方向一阶导数。

    Parameters
    ----------
    w : np.ndarray
        标量场。
    dy : float
        网格间距 [m]。

    Returns
    -------
    np.ndarray
        ∂w/∂y 的离散估计。

    数学公式
    --------
    - 内点： (w(i+1,j) - w(i-1,j)) / (2 dy)
    - 上边界： (w(1,j) - w(0,j)) / dy
    - 下边界： (w(-1,j) - w(-2,j)) / dy

    精度
    ----
    - 内点二阶，边界一阶。
    """
    gy = np.empty_like(w, dtype=np.float64)
    gy[1:-1, :] = (w[2:, :] - w[:-2, :]) / (2.0 * dy)
    gy[0, :] = (w[1, :] - w[0, :]) / dy
    gy[-1, :] = (w[-1, :] - w[-2, :]) / dy
    return gy


# -------------------------
# 主计算：Qx,Qy,R,q（允许分别用不同的预处理）
# -------------------------
def compute_Q_R_from_w(
    w: np.ndarray, a: float, E: float, nu: float, h: float, Gp: float, k: float
):
    """根据位移场 w 计算剪力 Qx、Qy 与基床反力 R。

    Parameters
    ----------
    w : np.ndarray
        位移场，尺寸 N×N（需要正方网格）。
    a : float
        物理域边长 [m]，假设均匀网格，dx = a/(N-1)。
    E : float
        弹性模量 [Pa]。
    nu : float
        泊松比 [-]。
    h : float
        板厚 [m]。
    Gp : float
        Pasternak 剪切参数 [Pa·m]。
    k : float
        Winkler 基床系数 [N/m³]。

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        Qx[N/m], Qy[N/m], R[N/m²]，均为 N×N。

    数学公式
    --------
    - 弯曲刚度：
        D = E h^3 / (12 (1 - ν^2))
    - 先计算 ∇²w，再取其梯度：
        Qx = - D ∂(∇²w)/∂x
        Qy = - D ∂(∇²w)/∂y                     （式 2 离散化）
    - 基床反力：
        R  = k w + G_p ∇² w                     （式 3）

    单位
    ----
    - Qx,Qy: N/m；R: N/m²（与 q 同量纲）。

    说明
    ----
    - 要求 w 为正方形网格（N×N），以便统一 dx=dy。
    - 数值噪声会在梯度运算中放大，建议在外部做适当平滑。
    """
    H, W = w.shape
    if H != W:
        raise ValueError(f"Expected square grid. Got H={H}, W={W}")
    N = H
    dx = a / (N - 1)
    D = E * (h ** 3) / (12.0 * (1.0 - nu ** 2))

    lap = laplacian_5pt(w, dx)
    dlap_dx = grad_central_x(lap, dx)
    dlap_dy = grad_central_y(lap, dx)

    Qx = -D * dlap_dx
    Qy = -D * dlap_dy
    R = k * w + Gp * lap
    return Qx, Qy, R


def compute_q_from_w(
    w: np.ndarray, a: float, E: float, nu: float, h: float, Gp: float, k: float
):
    """根据位移场 w 计算分布载荷 q。

    Parameters
    ----------
    w : np.ndarray
        位移场，尺寸 N×N。
    a : float
        物理域边长 [m]，dx = a/(N-1)。
    E, nu, h, Gp, k : see `compute_Q_R_from_w`.

    Returns
    -------
    np.ndarray
        q[N/m²]，尺寸 N×N。

    数学公式
    --------
    由薄板方程 (1)：
        q = D ∇⁴ w + G_p ∇² w + k w
    数值上：
        lap = ∇² w
        lap2 = ∇²(lap) = ∇⁴ w
        q = D * lap2 + G_p * lap + k * w

    稳定性
    ------
    - ∇⁴ 对高频噪声极敏感，故建议**强通道**先做更强平滑/低通/下采样。
    """
    H, W = w.shape
    if H != W:
        raise ValueError(f"Expected square grid. Got H={H}, W={W}")
    N = H
    dx = a / (N - 1)
    D = E * (h ** 3) / (12.0 * (1.0 - nu ** 2))

    lap = laplacian_5pt(w, dx)
    lap2 = laplacian_5pt(lap, dx)
    q = D * lap2 + Gp * lap + k * w
    return q


# -------------------------
# I/O
# -------------------------
def save_result(out_dir, forces_name, q_name, Qx, Qy, R, q, cfg, src_path):
    """将计算结果与元数据保存到磁盘。

    Parameters
    ----------
    out_dir : str
        输出目录（将自动创建）。
    forces_name : str
        三通道文件名，形如 "forces_3ch.npy"。
    q_name : str
        标量载荷文件名，形如 "q.npy"。
    Qx, Qy, R, q : np.ndarray
        对应力学量。
    cfg : dict
        运行配置（用于记录）。
    src_path : str
        输入文件源路径（记录追溯）。

    副作用
    ------
    - 保存:
        - forces_3ch.npy: (H,W,3) = [Qx, Qy, R]
        - q.npy:          (H,W)
        - meta.json:      元信息（单位、配置、源码路径等）
    """
    os.makedirs(out_dir, exist_ok=True)
    forces_3ch = np.stack([Qx, Qy, R], axis=-1).astype(np.float32)
    np.save(os.path.join(out_dir, forces_name), forces_3ch)
    np.save(os.path.join(out_dir, q_name), q.astype(np.float32))

    meta = {
        "units": {"w": "m", "Qx": "N/m", "Qy": "N/m", "R": "N/m^2", "q": "N/m^2"},
        "config": {
            "E": cfg.get("E"), "nu": cfg.get("nu"), "h": cfg.get("h"),
            "Gp": cfg.get("Gp"), "k": cfg.get("k"), "a": cfg.get("a"),
            # 轻通道
            "pre_smooth_sigma": cfg.get("pre_smooth_sigma", 0.0),
            "downsample_stride": cfg.get("downsample_stride", 1),
            "downsample_mode": cfg.get("downsample_mode", "mean"),
            "upsample_back": cfg.get("upsample_back", True),
            # 强通道（只作用于 q）
            "q_pre_smooth_sigma": cfg.get("q_pre_smooth_sigma", None),
            "q_downsample_stride": cfg.get("q_downsample_stride", None),
            "q_downsample_mode": cfg.get("q_downsample_mode", "mean"),
            "q_fft_lowpass_ratio": cfg.get("q_fft_lowpass_ratio", 0.0),
            "q_fft_lowpass_soft": cfg.get("q_fft_lowpass_soft", True),
            # 其他
            "enforce_zero_boundary": cfg.get("enforce_zero_boundary", False),
            "unit_check": cfg.get("unit_check", True),
            "w_scale_to_m": cfg.get("w_scale_to_m", 1.0),
        },
        "source": os.path.abspath(src_path),
        "forces_shape": list(forces_3ch.shape),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# -------------------------
# 单文件处理（双通道稳健流水线）
# -------------------------
def process_single(w_path: str, run_root: str, cfg: dict, params: tuple):
    """处理单个位移 .npy 文件并导出 Qx、Qy、R、q（固定尺寸输出）。

    Parameters
    ----------
    w_path : str
        输入位移文件路径（.npy, H×W, 2D）。
    run_root : str
        本次运行输出根目录。
    cfg : dict
        配置。
    params : tuple
        (E, nu, h, Gp, k, a) 物理参数。

    流水线
    ------
    1) 读取 w₀，并可选单位校验（|w|max<<1e-9 提醒可能未转米）。
    2) 单位归一：w = w₀ * w_scale_to_m。
    3) 轻通道（Qx,Qy,R）：轻平滑 → 可选置零边界 → 下采样 → 物理解算 → 最近邻上采样回原尺寸。
    4) 强通道（q）：更强平滑 → 可选频域低通 → 更大步长下采样 → 物理解算 → 最近邻上采样。
    5) 保存 forces_3ch.npy、q.npy 与 meta.json。

    Returns
    -------
    str or None
        输出目录；输入非法时返回 None。

    Notes
    -----
    - 轻/强通道仅影响**中间数值稳定性**，最终输出统一为 H×W。
    - “enforce_zero_boundary=True”将边框一圈置 0，可用于排除边缘伪影，
      但通常不建议启用（可能引入非物理梯度）。
    """
    E, nu, h, Gp, k, a = params
    w0 = np.load(w_path)
    if w0.ndim != 2:
        print(f"[Skip] Not 2D array: {w_path} shape={w0.shape}")
        return None

    H0, W0 = w0.shape
    if bool(cfg.get("unit_check", True)):
        mx = float(np.nanmax(np.abs(w0)))
        if mx < 1e-9:
            print(f"[Warn] |w|max={mx:.3e} m << 1e-9 ? 检查 w 的单位（应为米）: {w_path}")

    # 统一单位（若需要）
    w_scale = float(cfg.get("w_scale_to_m", 1.0) or 1.0)
    if w_scale != 1.0:
        w0 = w0.astype(np.float64) * w_scale

    # === 轻通道：用于 Qx, Qy, R ===
    w_lr = w0.copy()
    # 平滑
    sigma_lr = float(cfg.get("pre_smooth_sigma", 0.0) or 0.0)
    if sigma_lr > 0:
        w_lr = gaussian_blur(w_lr, sigma_lr)
    # 边界置零（一般不建议开）
    if bool(cfg.get("enforce_zero_boundary", False)):
        w_lr = w_lr.copy()
        w_lr[0, :] = w_lr[-1, :] = w_lr[:, 0] = w_lr[:, -1] = 0.0
    # 下采样
    stride_lr = int(cfg.get("downsample_stride", 1) or 1)
    mode_lr = str(cfg.get("downsample_mode", "mean")).lower()
    w_lr_small = downsample(w_lr, stride_lr, mode=mode_lr)
    # 计算
    Qx_s, Qy_s, R_s = compute_Q_R_from_w(w_lr_small, a, E, nu, h, Gp, k)
    # 回到原尺寸（固定输出 H0×W0）
    Qx = upsample_nearest(Qx_s, (H0, W0))
    Qy = upsample_nearest(Qy_s, (H0, W0))
    R  = upsample_nearest(R_s,  (H0, W0))

    # === 强通道：仅用于 q ===
    w_hr = w0.copy()
    # 更强高斯
    sigma_q = cfg.get("q_pre_smooth_sigma", None)
    sigma_q = float(sigma_q if (sigma_q is not None) else max(4.0, sigma_lr))
    if sigma_q > 0:
        w_hr = gaussian_blur(w_hr, sigma_q)
    # 频域低通（只对 q）
    lp_ratio = float(cfg.get("q_fft_lowpass_ratio", 0.0) or 0.0)
    if lp_ratio > 0:
        w_hr = fft_lowpass2d(w_hr, ratio=lp_ratio, soft=bool(cfg.get("q_fft_lowpass_soft", True)))
    # 下采样（q 通道可更粗）
    stride_q = cfg.get("q_downsample_stride", None)
    if stride_q is None:
        # 若未显式给出，用 max(5, stride_lr) 更稳
        stride_q = max(5, stride_lr)
    stride_q = int(stride_q)
    mode_q = str(cfg.get("q_downsample_mode", "mean")).lower()
    w_hr_small = downsample(w_hr, stride_q, mode=mode_q)
    q_s = compute_q_from_w(w_hr_small, a, E, nu, h, Gp, k)
    q = upsample_nearest(q_s, (H0, W0))  # 固定输出 H0×W0

    # 保存
    rel = os.path.splitext(os.path.basename(w_path))[0]
    out_dir = os.path.join(run_root, rel)
    save_result(
        out_dir=out_dir,
        forces_name=cfg.get("output_forces_name", "forces_3ch.npy"),
        q_name=cfg.get("output_q_name", "q.npy"),
        Qx=Qx, Qy=Qy, R=R, q=q, cfg=cfg, src_path=w_path
    )
    print(f"[OK] {w_path} -> {out_dir}")
    return out_dir


# -------------------------
# CLI
# -------------------------
def main():
    """提供命令行入口以批量或单文件执行反算流程。

    CLI
    ---
    --config <YAML>            : 必填，配置文件。
    --input <w.npy>            : 单文件模式（优先级高于 --input-dir）。
    --input-dir <dir>          : 批处理目录。
    --pattern <glob>           : 文件匹配后缀，默认 *.npy（仅后缀过滤）。
    --recursive / --no-recursive: 递归或不递归子目录（命令行覆盖配置）。
    --output-root <dir>        : 输出根目录；默认 cfg.output_root 或 ./forces_out。

    Config 关键字段
    --------------
    物理：
      E[Pa], nu[-], h[m], Gp[Pa·m], k[N/m³], a[m]
    单位 / 校验：
      unit_check[bool], w_scale_to_m[float]
    轻通道：
      pre_smooth_sigma, downsample_stride, downsample_mode, upsample_back
    强通道(q)：
      q_pre_smooth_sigma, q_downsample_stride, q_downsample_mode,
      q_fft_lowpass_ratio, q_fft_lowpass_soft
    其他：
      enforce_zero_boundary[bool], input_dir, input_glob, recursive

    行为
    ----
    - 若提供 --input，处理单文件并在 run_YYYY-mm-dd_HH-MM-SS 子目录输出。
    - 否则根据 input_dir+pattern 枚举批处理。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--input", default=None, help="single w.npy (H,W)")
    parser.add_argument("--input-dir", default=None, help="batch mode: dir of .npy")
    parser.add_argument("--pattern", default=None, help="glob like *.npy")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 基本参数
    E = float(cfg.get("E", 2.0e9))
    nu = float(cfg.get("nu", 0.30))
    h = float(cfg.get("h", 1.0e-3))
    Gp = float(cfg.get("Gp", 1.0e5))
    k = float(cfg.get("k", 5.0e6))
    a = float(cfg.get("a", 1.0))
    params = (E, nu, h, Gp, k, a)

    out_root = args.output_root or cfg.get("output_root", "./forces_out")
    run_dir = os.path.join(out_root, timestamp())
    os.makedirs(run_dir, exist_ok=True)

    # 单文件
    if args.input:
        process_single(args.input, run_dir, cfg, params)
        print(f"[DONE] Single file -> {run_dir}")
        return

    # 批处理
    input_dir = args.input_dir or cfg.get("input_dir", "./w_dir")
    pattern = args.pattern or cfg.get("input_glob", "*.npy")
    recursive = True if args.recursive else (False if args.no_recursive else bool(cfg.get("recursive", True)))

    paths = list(iter_w_files(input_dir, pattern, recursive))
    if not paths:
        print(f"[Warn] No input files under {input_dir} (pattern={pattern})")
        return

    print(f"[Info] Found {len(paths)} files. Output -> {run_dir}")
    ok = 0
    for p in paths:
        try:
            process_single(p, run_dir, cfg, params)
            ok += 1
        except Exception as e:
            print(f"[Error] {p}: {e}")
    print(f"[DONE] {ok}/{len(paths)} processed. See: {run_dir}")


if __name__ == "__main__":
    main()
