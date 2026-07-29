import importlib.util
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODULE_PATH=ROOT/"simulation"/"03p_replay_fddp_single_step_mujoco.py"

def load_module():
    spec=importlib.util.spec_from_file_location("single_step_replay",MODULE_PATH)
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

class SingleStepReplayTests(unittest.TestCase):
    def test_reference_matches_single_step_contract(self):
        module=load_module()
        t,q,dq,com,phase=module.load_reference()
        self.assertEqual(q.shape,(t.size,6))
        self.assertEqual(dq.shape,(t.size,6))
        self.assertEqual(com.shape,(t.size,3))
        self.assertEqual(phase[0],"double_support")
        self.assertEqual(phase[20],"right_stance_left_swing")
        self.assertEqual(phase[-1],"double_support_landing")
        self.assertIn("tau_ref_usage", MODULE_PATH.read_text())

if __name__=="__main__":
    unittest.main()
