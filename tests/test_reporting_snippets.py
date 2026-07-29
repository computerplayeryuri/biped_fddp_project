import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "reporting" / "generate_overleaf_evidence_snippets.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_overleaf_evidence_snippets", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvidenceSnippetGenerationTests(unittest.TestCase):
    def test_build_context_extracts_current_c2_and_hardware_evidence(self):
        module = load_module()

        context = module.build_context(ROOT)

        self.assertEqual(context["c2_contact_demo"]["matched_events"], 2)
        self.assertEqual(context["c2_contact_demo"]["scheduled_events"], 2)
        self.assertAlmostEqual(context["c2_contact_demo"]["max_abs_phase_offset_ms"], 35.0)
        self.assertTrue(context["hardware_runtime"]["all_loadability_tests_passed"])
        self.assertEqual(context["stage8_scripted_locomotion"]["matched_contact_events"], 4)
        self.assertAlmostEqual(context["stage8_scripted_locomotion"]["forward_displacement_m"], 0.0136, places=3)
        self.assertEqual(context["missing_locomotion_metrics"][0], "walking duration over 10/30/60 s")

    def test_render_snippet_contains_overleaf_ready_c2_runtime_and_figure_refs(self):
        module = load_module()
        context = module.build_context(ROOT)

        tex = module.render_latex(context)

        self.assertIn("\\section{C2 Contact-Phase Synchronisation}", tex)
        self.assertIn("matched 2/2 scheduled contact events", tex)
        self.assertIn("\\section{Hardware Runtime Export}", tex)
        self.assertIn("hardware/controller_runtime/controller_runtime.py", tex)
        self.assertIn("\\section{Stage 8 Scripted Alternating-Step Locomotion Preview}", tex)
        self.assertIn("matched 4/4 scheduled contact events", tex)
        self.assertIn("results/figures/c2_contact_switching_demo_phase_offset.png", tex)
        self.assertIn("results/figures/stage8_scripted_alternating_step_body.png", tex)
        self.assertIn("walking duration over 10/30/60 s", tex)

    def test_main_writes_snippet_and_manifest(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            module.main(["--project-root", str(ROOT), "--output-dir", str(out_dir)])

            tex_path = out_dir / "overleaf_current_evidence_update.tex"
            manifest_path = out_dir / "overleaf_current_evidence_manifest.txt"
            self.assertTrue(tex_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertIn("C2 contact-phase", tex_path.read_text(encoding="utf-8"))
            self.assertIn("c2_contact_switching_demo_phase_offset.png", manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
