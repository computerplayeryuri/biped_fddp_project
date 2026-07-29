# Thesis Result Archive Policy

All simulation result artifacts remain in the project workspace as the source of truth and are mirrored to:

`D:\毕业论文\机器人控制流程图\FDDP仿真结果归档`

Included trees:

- `results/`
- `transfer/trajectories/`
- `docs/`

After final verification of every future experiment, run from Windows PowerShell:

```powershell
& "\\wsl$\Ubuntu-22.04\home\zhijian\biped_fddp_project\tools\sync_thesis_results_to_d_drive.ps1"
```

The script copies without deleting existing destination files, verifies every copied file by SHA-256, and regenerates `SHA256_MANIFEST.csv` in the archive root.

This archive policy does not broaden the claim boundary: results are simulation validation and runtime-loadability evidence unless a later, separately approved hardware phase produces real-hardware evidence.
