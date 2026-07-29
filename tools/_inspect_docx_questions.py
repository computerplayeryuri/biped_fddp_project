from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document


def main() -> None:
    path = Path(sys.argv[1])
    document = Document(path)
    payload = {
        "path": str(path),
        "paragraphs": [
            {"index": index, "style": paragraph.style.name, "text": paragraph.text}
            for index, paragraph in enumerate(document.paragraphs)
        ],
        "tables": [
            {
                "index": table_index,
                "rows": [
                    [cell.text for cell in row.cells]
                    for row in table.rows
                ],
            }
            for table_index, table in enumerate(document.tables)
        ],
        "sections": len(document.sections),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
