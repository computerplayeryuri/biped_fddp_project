from __future__ import annotations

import re
import sys
from pathlib import Path


def brace_balance(text: str) -> int:
    balance = 0
    for index, char in enumerate(text):
        escaped = index > 0 and text[index - 1] == "\\"
        if escaped:
            continue
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
            if balance < 0:
                raise RuntimeError("closing brace appears before matching opening brace")
    return balance


def main() -> None:
    guide_path = Path(sys.argv[1])
    v4_path = Path(sys.argv[2])
    overleaf_root = Path(sys.argv[3])
    guide = guide_path.read_text(encoding="utf-8")
    v4 = v4_path.read_text(encoding="utf-8")

    if brace_balance(guide) != 0:
        raise RuntimeError("learning guide has unbalanced braces")
    if brace_balance(v4) != 0:
        raise RuntimeError("v4 has unbalanced braces")

    labels = set(re.findall(r"\\label\{([^}]+)\}", guide))
    refs = set()
    for pattern in (
        r"\\ref\*?\{([^}]+)\}",
        r"\\hyperref\[([^\]]+)\]",
        r"\\(?:c|C)ref\{([^}]+)\}",
    ):
        refs.update(re.findall(pattern, guide))
    missing_refs = sorted(refs - labels)
    if missing_refs:
        raise RuntimeError(f"undefined internal references: {missing_refs}")

    duplicate_labels = sorted(
        label for label in labels if guide.count(r"\label{" + label + "}") != 1
    )
    if duplicate_labels:
        raise RuntimeError(f"duplicate labels: {duplicate_labels}")

    question_labels = {f"q{index}" for index in range(1, 21)}
    if not question_labels.issubset(labels):
        raise RuntimeError(f"missing question labels: {sorted(question_labels - labels)}")

    bib_text = (overleaf_root / "sample.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    citation_keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", guide):
        citation_keys.update(key.strip() for key in group.split(","))
    missing_citations = sorted(citation_keys - bib_keys)
    if missing_citations:
        raise RuntimeError(f"missing bibliography entries: {missing_citations}")

    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", guide)
    missing_figures = sorted(name for name in figures if not (overleaf_root / name).exists())
    if missing_figures:
        raise RuntimeError(f"missing figures: {missing_figures}")

    unsafe_code_args = re.findall(r"\\code\{([^}]*(?<!\\)_[^}]*)\}", guide)
    if unsafe_code_args:
        raise RuntimeError(f"unescaped underscores in code macro: {unsafe_code_args}")

    if "project_complete_learning_guide_zh" in v4:
        raise RuntimeError("standalone learning guide was included by v4")

    required_v4_phrases = (
        "independently solved left- and right-swing FDDP steps",
        "The correct RQ1 answer is therefore positive",
        "fails the current friction-utilisation and CoP-margin gates",
        "Do not generalise the tested C1/two-step result",
    )
    missing_v4 = [phrase for phrase in required_v4_phrases if phrase not in v4]
    if missing_v4:
        raise RuntimeError(f"required v4 updates missing: {missing_v4}")

    forbidden_v4_phrases = (
        "The current work is limited to double-support CoM shifts and scripted alternating steps.",
        "Walking-gait validation & Not completed",
        "Do not claim that C1 works for walking",
    )
    retained = [phrase for phrase in forbidden_v4_phrases if phrase in v4]
    if retained:
        raise RuntimeError(f"stale v4 phrases retained: {retained}")

    print(
        "static TeX validation passed: "
        f"labels={len(labels)}, refs={len(refs)}, questions=20, "
        f"citations={len(citation_keys)}, figures={len(figures)}"
    )


if __name__ == "__main__":
    main()
