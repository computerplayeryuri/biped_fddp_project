from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATEXMKRC = PROJECT_ROOT / "docs" / "overleaf" / "latexmkrc"


class OverleafLearningGuideBuildTest(unittest.TestCase):
    def test_latexmkrc_forces_xelatex_even_when_pdf_mode_is_selected(self):
        config = LATEXMKRC.read_text(encoding="utf-8")

        self.assertIn("$pdf_mode = 5;", config)
        self.assertIn("$pdflatex = 'xelatex %O %S';", config)
        self.assertIn("$xelatex = 'xelatex %O %S';", config)


if __name__ == "__main__":
    unittest.main()
