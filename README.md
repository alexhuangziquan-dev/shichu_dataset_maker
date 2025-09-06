
# Pasternak Force Solver v4 (outputs q and visualizes it)

Now outputs **both**:
- `forces_3ch.npy` of shape (H,W,3): channels `[Qx, Qy, R]`
- `q.npy` of shape (H,W): external load computed by `q = D * Laplacian(Laplacian(w)) + Gp * Laplacian(w) + k * w`

Batch compute and batch visualization supported; timestamped run folders.

## Compute

Single file:
```bash
python tools/force_from_shape.py --config configs/force_from_shape.yaml --input ./w.npy
```

Batch directory:
```bash
python tools/force_from_shape.py --config configs/force_from_shape.yaml --input-dir ./w_dir --recursive
# optional: --pattern *.npy  --output-root ./forces_out
```

Output layout:
```
<output_root>/run_YYYY-MM-DD_HH-MM-SS/<stem>/
  forces_3ch.npy   # (H,W,3) [Qx,Qy,R]
  q.npy            # (H,W)
  meta.json
```

## Visualize

```bash
# visualize everything under a run folder (both forces and q)
python tools/visualize_forces.py --input ./forces_out/run_YYYY-MM-DD_HH-MM-SS --recursive --keep-rel --outroot ./viz_out --with-mag
```
The script detects array type by shape: (H,W,3) -> forces, (H,W) -> q.

## Notes
- Units: Qx,Qy [N/m], R [N/m^2], q [N/m^2].
- Set `a` so that `dx=a/(N-1)` matches physical spacing.
- If w edges are not zero, consider `enforce_zero_boundary: true`.

## Requirements
```
numpy
pyyaml
matplotlib
```
