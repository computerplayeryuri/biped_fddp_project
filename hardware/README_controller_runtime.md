# Hardware Controller Runtime

This folder contains a MuJoCo-free Python runtime for loading the three thesis
controllers on a PC/ROS2-side hardware pipeline.

## Controllers

- C0: FDDP joint-reference playback, no online correction.
- C1: fixed-phase FIR-QP-MPC correction, outputs q_cmd = q_ref + delta_q.
- C2: C1 plus toe/heel contact-event phase synchronisation.

## Minimal usage

```python
from hardware.controller_runtime import make_controller, ControllerMeasurement

controller = make_controller("C2", direction="forward")
measurement = ControllerMeasurement(
    time_s=0.40,
    root_pitch_rad=0.0,
    root_pitch_rate_rad_s=0.0,
    left_toe_contact=True,
    left_heel_contact=True,
    right_toe_contact=True,
    right_heel_contact=True,
)
out = controller.update(measurement)
q_cmd_rad = out["q_cmd_rad"]
```

## Safety

The runtime clips joint commands to `config/robot_limits.yaml` and rate-limits
commands using `config/actuator_model.yaml`. It does not communicate with servos
by itself; a hardware bridge must still implement serial communication,
timeouts, emergency stop, servo fault handling, and calibration.
