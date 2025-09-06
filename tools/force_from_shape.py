import argparse, os, yaml, json
from datetime import datetime
import numpy as np

"""从位移场 w 反算力学场（Google 风格文档）。

给定薄板在 Pasternak 地基上的位移场 `w(x, y)`（二维数组，单位 m），
计算并输出以下场量：
- `forces_3ch`：形状 (H, W, 3)，通道 [Qx, Qy, R]
  - Qx, Qy：剪力结果/单位长度（N/m），定义为 `-D * ∂(∇²w)/∂x(y)`
  - R：地基竖向反力/压力（N/m²），`k*w + Gp*∇²w`
- `q`：等效外载荷（N/m²），`D*∇⁴w + Gp*∇²w + k*w`

输出结构：
    <output_root>/run_YYYY-MM-DD_HH-MM-SS/<case>/
        forces_3ch.npy   # (H,W,3) [Qx,Qy,R]
        q.npy            # (H,W)
        meta.json        # 元信息（参数、时间、来源）

Notes:
    - 网格步长 `dx = a/(N-1)`，其中 N=H=W。
    - 可以选择在计算前强制 w 的四边为 0（enforce_zero_boundary）。
"""


def load_config(path):
    """读取 YAML 配置。

    Args:
        path (str): 配置文件路径。

    Returns:
        dict: 解析后的配置字典。
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def timestamp():
    """生成时间戳子目录名（run_YYYY-MM-DD_HH-MM-SS）。

    Returns:
        str: 时间戳目录名。
    """
    return datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")


def compute_fields_from_shape(w, a, E, nu, h, Gp, k):
    """根据位移场 `w` 计算 Qx、Qy、R、q。

    数学定义：
        D = E*h^3 / (12*(1-nu^2))
        lap   = ∇²w
        lap2  = ∇²(∇²w) = ∇⁴w
        Qx,Qy = -D * ∂(lap)/∂x(y)
        R     = k*w + Gp*lap
        q     = D*lap2 + Gp*lap + k*w

    Args:
        w (np.ndarray): 位移场，形状 (H, W)，单位 m。
        a (float): 板边长（m）。用于计算 dx = a/(N-1)。
        E (float): 弹性模量（Pa）。
        nu (float): 泊松比（-）。
        h (float): 板厚（m）。
        Gp (float): Pasternak 剪切参数（Pa·m）。
        k (float): Winkler 基床系数（N/m^3）。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Qx (H,W), Qy (H,W) 单位 N/m；R (H,W) 单位 N/m²；q (H,W) 单位 N/m²。

    Raises:
        ValueError: 当输入网格非正方形 (H != W) 时。
    """
    H, W = w.shape
    if H != W:
        raise ValueError(f"Expected square grid (H==W). Got H={H}, W={W}")
    N = H

    # Bending stiffness
    D = E * h**3 / (12.0 * (1.0 - nu**2))

    # Physical grid spacing
    dx = a / (N - 1)
    dy = dx

    # First Laplacian
    wx, wy = np.gradient(w, dx, dy, edge_order=2)
    wxx, _  = np.gradient(wx, dx, dy, edge_order=2)
    _,  wyy = np.gradient(wy, dx, dy, edge_order=2)
    lap = wxx + wyy

    # Shear resultants
    lapx, lapy = np.gradient(lap, dx, dy, edge_order=2)
    Qx = -D * lapx
    Qy = -D * lapy

    # Foundation reaction
    R = k * w + Gp * lap

    # Biharmonic (Laplacian of Laplacian)
    lapx2, lapy2 = np.gradient(lap, dx, dy, edge_order=2)
    lapxx, _     = np.gradient(lapx2, dx, dy, edge_order=2)
    _,     lapyy = np.gradient(lapy2, dx, dy, edge_order=2)
    lap2 = lapxx + lapyy

    # External load per unit area
    q = D * lap2 + Gp * lap + k * w

    return Qx, Qy, R, q


def save_result(out_dir, forces_name, q_name, Qx, Qy, R, q, cfg, src_path):
    """保存计算结果到指定目录（forces_3ch.npy、q.npy、meta.json）。

    Args:
        out_dir (str): 输出目录。
        forces_name (str): 力场文件名（默认 `forces_3ch.npy`）。
        q_name (str): 外载荷文件名（默认 `q.npy`）。
        Qx, Qy (np.ndarray): 剪力结果（H,W），单位 N/m。
        R (np.ndarray): 地基反力（H,W），单位 N/m²。
        q (np.ndarray): 外载荷（H,W），单位 N/m²。
        cfg (dict): 配置字典，用于写入元信息。
        src_path (str): 源位移文件路径。

    Returns:
        tuple[str, str]: (forces_path, q_path) 两个输出文件的路径。
    """
    os.makedirs(out_dir, exist_ok=True)
    # forces
    forces_3ch = np.stack([Qx, Qy, R], axis=-1).astype(np.float32)
    forces_path = os.path.join(out_dir, forces_name)
    np.save(forces_path, forces_3ch)
    # q
    q_path = os.path.join(out_dir, q_name)
    np.save(q_path, q.astype(np.float32))
    # meta
    meta = {
        "source": src_path,
        "forces_path": forces_path,
        "q_path": q_path,
        "forces_shape": list(forces_3ch.shape),
        "q_shape": list(q.shape),
        "E": float(cfg.get("E", 2e9)),
        "nu": float(cfg.get("nu", 0.3)),
        "h": float(cfg.get("h", 1e-3)),
        "Gp": float(cfg.get("Gp", 1e5)),
        "k": float(cfg.get("k", 5e6)),
        "a": float(cfg.get("a", 1.0)),
        "enforce_zero_boundary": bool(cfg.get("enforce_zero_boundary", False)),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # optional legacy
    if bool(cfg.get("save_individual_channels", False)):
        np.save(os.path.join(out_dir, "Qx.npy"), Qx)
        np.save(os.path.join(out_dir, "Qy.npy"), Qy)
        np.save(os.path.join(out_dir, "R.npy"),  R)
        np.savez(os.path.join(out_dir, "forces.npz"), Qx=Qx, Qy=Qy, R=R)
    return forces_path, q_path


def process_single(w_path, run_root, cfg, params):
    """处理单个位移文件并保存对应输出。

    Args:
        w_path (str): 输入位移 `w.npy` 路径（(H,W)）。
        run_root (str): 本次 run 的根输出目录。
        cfg (dict): 配置字典。
        params (tuple): (E, nu, h, Gp, k, a)。

    Returns:
        tuple[str, str] | None: (forces_path, q_path)。若输入不是 2D 数组则返回 None。
    """
    E, nu, h, Gp, k, a = params
    w = np.load(w_path)
    if w.ndim != 2:
        print(f"[Skip] Not 2D array: {w_path} shape={w.shape}")
        return None
    if bool(cfg.get("enforce_zero_boundary", False)):
        w[0,:]=w[-1,:]=w[:,0]=w[:,-1]=0.0
    Qx, Qy, R, q = compute_fields_from_shape(w, a, E, nu, h, Gp, k)
    rel = os.path.splitext(os.path.basename(w_path))[0]
    out_dir = os.path.join(run_root, rel)
    forces_name = cfg.get("output_forces_name", "forces_3ch.npy")
    q_name = cfg.get("output_q_name", "q.npy")
    fpath, qpath = save_result(out_dir, forces_name, q_name, Qx, Qy, R, q, cfg, w_path)
    print(f"[OK] {w_path} -> {fpath} ; {qpath}")
    return fpath, qpath


def iter_w_files(root_dir, pattern, recursive):
    """生成器：遍历 root_dir 下符合模式的 .npy 位移文件。

    Args:
        root_dir (str): 输入根目录。
        pattern (str): 匹配模式（如 `*.npy`），当前实现支持前缀 `*.` 的简单后缀匹配。
        recursive (bool): 是否递归子目录。

    Yields:
        str: 匹配到的文件的绝对路径。
    """
    if recursive:
        for dirpath, _, files in os.walk(root_dir):
            for fn in files:
                if fn.lower().endswith(".npy") and _match(fn, pattern):
                    yield os.path.join(dirpath, fn)
    else:
        for fn in os.listdir(root_dir):
            p = os.path.join(root_dir, fn)
            if os.path.isfile(p) and fn.lower().endswith(".npy") and _match(fn, pattern):
                yield p


def _match(filename, pattern):
    """简单的通配匹配：支持 `*` 与 `*.xxx`，否则退化为子串包含。

    Args:
        filename (str): 文件名。
        pattern (str): 模式字符串。

    Returns:
        bool: 是否匹配。
    """
    # simple glob: '*' wildcard and '*.xxx'
    if pattern in (None, "", "*", "*.npy"):
        return True
    if pattern.startswith("*."):
        return filename.endswith(pattern[1:])
    return pattern in filename


def main():
    """命令行入口：单文件/批量模式计算并输出 Qx、Qy、R、q。"""
    parser = argparse.ArgumentParser(description="Compute (Qx,Qy,R) AND q from w. Supports single or batch mode.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--input", default=None, help="override single input file (H,W)")
    parser.add_argument("--input-dir", default=None, help="override input_dir for batch mode")
    parser.add_argument("--pattern", default=None, help="override input_glob, e.g. *.npy")
    parser.add_argument("--recursive", action="store_true", help="override recursive to True")
    parser.add_argument("--no-recursive", action="store_true", help="override recursive to False")
    parser.add_argument("--output-root", default=None, help="override output_root")
    args = parser.parse_args()

    cfg = load_config(args.config)
    E  = float(cfg.get("E", 2.0e9))
    nu = float(cfg.get("nu", 0.30))
    h  = float(cfg.get("h", 1.0e-3))
    Gp = float(cfg.get("Gp", 1.0e5))
    k  = float(cfg.get("k", 5.0e6))
    a  = float(cfg.get("a", 1.0))
    params = (E, nu, h, Gp, k, a)

    out_root = args.output_root or cfg.get("output_root", "./forces_out")
    run_dir = os.path.join(out_root, timestamp())
    os.makedirs(run_dir, exist_ok=True)

    # Single-file mode
    if args.input:
        process_single(args.input, run_dir, cfg, params)
        print(f"[DONE] Single file. Outputs under: {run_dir}")
        return

    # Batch mode
    input_dir = args.input_dir or cfg.get("input_dir", "./w_dir")
    pattern = args.pattern or cfg.get("input_glob", "*.npy")
    recursive = cfg.get("recursive", True)
    if args.recursive: recursive = True
    if args.no_recursive: recursive = False

    count = 0
    for w_path in iter_w_files(input_dir, pattern, recursive):
        base = os.path.basename(w_path).lower()
        if base in (cfg.get("output_forces_name", "forces_3ch.npy").lower(),
                    cfg.get("output_q_name", "q.npy").lower()):
            continue
        try:
            process_single(w_path, run_dir, cfg, params)
            count += 1
        except Exception as e:
            print(f"[Error] {w_path}: {e}")
    print(f"[DONE] Batch processed {count} files. Outputs under: {run_dir}")


if __name__ == "__main__":
    main()
