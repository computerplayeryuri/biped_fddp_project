# PPT参考依据（可复制）

## FDDP与多接触最优控制

1. C. Mastalli et al., “Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control,” IEEE ICRA, 2020, pp. 2536–2542, doi: 10.1109/ICRA40945.2020.9196673.

2. C. Mastalli et al., “A Feasibility-Driven Approach to Control-Limited DDP,” Autonomous Robots, vol. 46, pp. 985–1005, 2022, doi: 10.1007/s10514-022-10061-w.

## MPC与经验响应预测

3. J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, Model Predictive Control: Theory, Computation, and Design, 2nd ed., Nob Hill Publishing, 2017.

4. C. R. Cutler and B. L. Ramaker, “Dynamic Matrix Control—A Computer Control Algorithm,” Joint Automatic Control Conference, 1980.

## 动力学与仿真

5. J. Carpentier et al., “The Pinocchio C++ Library—A Fast and Flexible Implementation of Rigid Body Dynamics Algorithms and Their Analytical Derivatives,” IEEE/SICE SII, 2019.

6. E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A Physics Engine for Model-Based Control,” IEEE/RSJ IROS, 2012, pp. 5026–5033, doi: 10.1109/IROS.2012.6386109.

## PPT中的推荐引用方式

- FDDP页面：`Method based on Crocoddyl/FDDP [1], [2].`
- MPC页面：`Receding-horizon formulation follows standard MPC [3]; the empirical FIR prediction follows the Dynamic Matrix Control principle [4].`
- 仿真页面：`Independent replay is conducted in MuJoCo [6], using a separately audited MJCF model.`

