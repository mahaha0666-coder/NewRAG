"""Validate and merge per-view CAD evidence annotations into one JSONL index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mechrag.evidence import EvidenceStore, EvidenceUnit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", help="Input JSONL, one feature/view record per line")
    parser.add_argument("output", help="Output JSONL with cross-view duplicates merged")
    args = parser.parse_args()

    units = []
    with Path(args.annotations).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                units.append(EvidenceUnit.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid annotation at line {line_number}") from exc

    store = EvidenceStore(
        units,
        expected_part_ids={unit.part_id for unit in units},
        universe_complete=True,
    )
    store.save_jsonl(args.output)
    print(
        json.dumps(
            {
                "input_records": len(units),
                "deduplicated_evidence_units": len(store.units),
                "parts": len(store.part_ids),
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
