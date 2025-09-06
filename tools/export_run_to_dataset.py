# -*- coding: utf-8 -*-
"""将某次 run 目录导出为扁平数据集（Google 风格文档）。

本脚本会递归扫描指定的 run 目录，分别收集：
- `forces_3ch.npy`（形状 (H, W, 3)，通道 [Qx, Qy, R]）
- `q.npy`（形状 (H, W)，外载荷场）

并将它们各自导出到目标目录下的 `f/` 与 `q/` 子目录中，按顺序重命名为
`000000.npy, 000001.npy, ...`，同时在输出根目录生成清单：
`manifest_f.csv` 与 `manifest_q.csv`，用于记录“新文件名 <-> 源相对路径”。

Example:
    # PowerShell 示例
    python tools\\export_run_to_dataset.py --run .\\forces_out\\run_2025-09-06_18-58-50 --out .\\dataset

Notes:
    - 导出次序**稳定且可复现**：按源文件相对 run 根目录的相对路径排序（大小写不敏感）。
    - forces 与 q **分别**独立编号（各自从 --start 开始）。
"""

import os
import argparse
import shutil
import csv


def find_files(run_dir, target_name):
    """在 run 目录递归查找指定文件名（大小写不敏感），并按相对路径排序返回。

    Args:
        run_dir (str): run 根目录路径，例如 `./forces_out/run_YYYY-MM-DD_HH-MM-SS`。
        target_name (str): 目标文件名（如 `forces_3ch.npy` 或 `q.npy`）。

    Returns:
        list[str]: 匹配文件的**绝对路径**列表，按“相对 run 根”的路径字典序（大小写不敏感）排序。
    """
    t_low = target_name.lower()
    hits = []
    for dirpath, _, files in os.walk(run_dir):
        for fn in files:
            if fn.lower() == t_low:
                hits.append(os.path.join(dirpath, fn))
    # 以相对路径排序，确保不同平台下一致；统一分隔符避免排序差异
    hits.sort(key=lambda p: os.path.relpath(os.path.abspath(p), os.path.abspath(run_dir)).lower().replace("\\", "/"))
    return hits


def ensure_dir(p):
    """确保目录存在（不存在则创建）。

    Args:
        p (str): 目录路径。
    """
    os.makedirs(p, exist_ok=True)


def export_group(paths, out_dir, start_idx=0, width=6, move=False, dry=False, run_dir=None, manifest_path=None):
    """导出一组文件到指定目录，并按序重命名（如 000000.npy）。

    Args:
        paths (list[str]): 待导出的源文件绝对路径列表。
        out_dir (str): 目标输出目录。
        start_idx (int, optional): 起始编号。默认为 0。
        width (int, optional): 数字零填充宽度。默认为 6（即 000000.npy）。
        move (bool, optional): True 则移动文件；False 则复制文件。默认 False。
        dry (bool, optional): True 则仅打印将执行的操作，不实际写文件。默认 False。
        run_dir (str, optional): run 根目录，用于在清单中记录相对路径。
        manifest_path (str, optional): 清单 CSV 输出路径。

    Returns:
        int: 实际导出的文件数量。
    """
    ensure_dir(out_dir)
    records = []  # 记录 (new_name, source_rel)
    idx = start_idx
    for src in paths:
        new_name = f"{idx:0{width}d}.npy"
        dst = os.path.join(out_dir, new_name)
        src_abs = os.path.abspath(src)
        src_rel = os.path.relpath(src_abs, os.path.abspath(run_dir)) if run_dir else src_abs

        if dry:
            action = "[DRY] move" if move else "[DRY] copy"
            print(f"{action}: {src_rel} -> {os.path.relpath(dst, os.path.abspath(out_dir))}")
        else:
            if move:
                shutil.move(src_abs, dst)
            else:
                shutil.copy2(src_abs, dst)

        records.append((new_name, src_rel))
        idx += 1

    # 写出清单 CSV
    if manifest_path:
        ensure_dir(os.path.dirname(manifest_path))
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["new_name", "source_rel"])
            w.writerows(records)

    return len(records)


def main():
    """命令行入口：将 run 目录导出为扁平数据集。"""
    ap = argparse.ArgumentParser(description="将某个 run 目录导出为数据集（扁平化，分别输出到 out/f 与 out/q，并按序重命名）。")
    ap.add_argument("--run", required=True, help="run 目录路径，例如：./forces_out/run_YYYY-MM-DD_HH-MM-SS")
    ap.add_argument("--out", required=True, help="输出根目录；会在其下创建子目录 'f' 与 'q'")
    ap.add_argument("--forces-name", default="forces_3ch.npy", help="forces 文件名（(H,W,3)）缺省为 forces_3ch.npy")
    ap.add_argument("--q-name", default="q.npy", help="q 文件名（(H,W)）缺省为 q.npy")
    ap.add_argument("--start", type=int, default=0, help="编号起始值（对 f 与 q 各自独立生效）")
    ap.add_argument("--width", type=int, default=6, help="文件名数字零填充宽度（默认 6 -> 000000.npy）")
    ap.add_argument("--move", action="store_true", help="将文件移动到目标处（默认复制）")
    ap.add_argument("--dry-run", action="store_true", help="仅演示（打印）将要执行的操作，不实际写文件")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    out_root = os.path.abspath(args.out)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"[Error] 未找到 run 目录：{run_dir}")

    # 收集 forces 与 q 文件
    forces_list = find_files(run_dir, args.forces_name)
    q_list = find_files(run_dir, args.q_name)
    print(f"[Info] 在 {run_dir} 下找到 {len(forces_list)} 个 forces 文件 与 {len(q_list)} 个 q 文件")

    # 准备输出目录
    out_f = os.path.join(out_root, "f")
    out_q = os.path.join(out_root, "q")
    ensure_dir(out_f)
    ensure_dir(out_q)

    # 导出 forces
    mf_f = os.path.join(out_root, "manifest_f.csv")
    n_forces = export_group(
        forces_list,
        out_f,
        start_idx=args.start,
        width=args.width,
        move=args.move,
        dry=args.dry_run,
        run_dir=run_dir,
        manifest_path=mf_f,
    )
    print(f"[OK] forces 导出到 -> {out_f} (数量={n_forces}, 起始={args.start}, 宽度={args.width})")
    print(f"[OK] forces 清单：{mf_f}")

    # 导出 q
    mf_q = os.path.join(out_root, "manifest_q.csv")
    n_q = export_group(
        q_list,
        out_q,
        start_idx=args.start,
        width=args.width,
        move=args.move,
        dry=args.dry_run,
        run_dir=run_dir,
        manifest_path=mf_q,
    )
    print(f"[OK] q 导出到 -> {out_q} (数量={n_q}, 起始={args.start}, 宽度={args.width})")
    print(f"[OK] q 清单：{mf_q}")

    if args.dry_run:
        print("[DONE] 预演完成（未写任何文件）。")
    else:
        print("[DONE] 导出完成。")


if __name__ == "__main__":
    main()
