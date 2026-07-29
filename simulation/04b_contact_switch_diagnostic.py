import mujoco

from common import MODEL_PATH, ContactDebouncer, contact_sensor_values

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Use the first keyframe (standing) defined in biped_planar.xml.
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

debouncer = ContactDebouncer()
control_dt = 0.020
next_print_time = 0.0

print("\n===== CONTACT SWITCH DIAGNOSTIC =====")
print("Expected final state: all toe/heel switches = 1")
print("Expected final state: left_full_support = 1, right_full_support = 1\n")

while data.time < 0.20:
    mujoco.mj_step(model, data)

    if data.time + 1e-12 >= next_print_time:
        raw = contact_sensor_values(model, data)
        switches = debouncer.update(raw)
        print(
            f"t={data.time:5.3f} s | "
            f"raw=[{raw['left_toe_touch']:.3f}, "
            f"{raw['left_heel_touch']:.3f}, "
            f"{raw['right_toe_touch']:.3f}, "
            f"{raw['right_heel_touch']:.3f}] | "
            f"switches={switches}"
        )
        next_print_time += control_dt

print("\nCONTACT SWITCH DIAGNOSTIC FINISHED")
