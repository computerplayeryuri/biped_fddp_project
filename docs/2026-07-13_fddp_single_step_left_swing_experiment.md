# FDDP Single-Step Left-Swing Experiment Record (2026-07-13)

Scope: independent DS -> right-stance/left-swing SS -> DS landing. Not continuous walking. No modification to 03n, Stage 8, or existing replay. MuJoCo is position-command replay only; tau_ref is analysis only.

Commands:
- /home/zhijian/miniforge3/envs/fddp_biped/bin/python fddp/04a_solve_fddp_single_step_left_swing.py
- /home/zhijian/miniforge3/envs/fddp_biped/bin/python simulation/03p_replay_fddp_single_step_mujoco.py
- targeted tests: 11 passed; full suite: 26 passed.

FDDP gates: max gap=0; support translation=5.622e-6 m; support rotation=6.883e-8 rad; landing translation=7.659e-8 m; landing rotation=6.883e-8 rad; clearance=1.9999e-2 m; joint margin=1.1995e-2 rad; CoM +Y=4.7103e-3 m. All specified numerical export gates passed.

Caveat: solver_return_success=false after 500 iterations, stop=1.0416e-3. Do not claim first-order convergence success.

MuJoCo replay: proxy delay=4 ms, lag=8 ms, speed=20 rad/s, q_ref only. RMSE=3.962 deg; max pitch=7.207 deg; min height=0.28751 m; min contacts=4. Replay integrity passed.

Claim boundary: simulation validation + runtime-loadability only, not autonomous hardware walking.

Artifacts:
- transfer/trajectories/fddp_single_step_left_swing_ref.npz
- transfer/trajectories/fddp_single_step_left_swing_ref_summary.yaml
- results/logs/fddp_single_step_left_swing_diagnostics.csv/.yaml
- results/figures/fddp_single_step_left_swing_com_clearance.png
- results/logs/fddp_single_step_left_swing_mujoco_replay.csv/.yaml
- results/figures/fddp_single_step_left_swing_mujoco_replay.png
