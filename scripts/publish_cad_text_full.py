"""Update the paper table from complete local text baseline scores."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time


DATASETS = ("DeepCAD", "TriView2CAD", "FCM")
METHODS = ("OCR + BM25", "OCR + BGE-M3 (dense)")
METRICS = ("acc", "recall", "f1")
READER = "qwen3-vl-2b-q4"


def completed_scores(root, expected):
    scores = {}
    for dataset in DATASETS:
        path = root / f"text-full-{dataset}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        if len(records) != len(METHODS) or {row["method"] for row in records} != set(METHODS):
            raise ValueError(f"Unexpected methods in {path}")
        for row in records:
            if row["dataset"] != dataset or row["reader"] != READER or int(row["samples"]) != expected[dataset]:
                raise ValueError(f"Unexpected dataset coverage or reader in {path}")
            scores[dataset, row["method"]] = [f"{float(row[key]):.2f}" for key in METRICS]
    return scores


def publish_rows(paper, scores):
    lines = ["% Generated from verified full-coverage scores."]
    for method in METHODS:
        values = [value for dataset in DATASETS for value in scores.get((dataset, method), [""] * 3)]
        cells = [method + r"$^{\dagger}$", *values, "", "", ""]
        lines.append("    " + " & ".join(cells) + r" \\")
    path = paper / "text-full-rows.tex"
    content = "\n".join(lines) + "\n"
    if path.read_text(encoding="utf-8") == content:
        return False
    staging = path.with_suffix(".tmp")
    staging.write_text(content, encoding="utf-8")
    staging.replace(path)
    return True


def rebuild_pdf(paper, build):
    build.mkdir(parents=True, exist_ok=True)
    with (build / "publisher.log").open("wb") as stream:
        subprocess.run(["latexmk", "-pdf", f"-outdir={build}", "-interaction=nonstopmode",
                        "-halt-on-error", "-file-line-error", "main.tex"],
                       cwd=paper, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for filename in ("main.pdf", "main-experiments.pdf"):
        try:
            shutil.copyfile(build / "main.pdf", paper / filename)
        except PermissionError:
            print(f"{filename} is open. Compiled PDF: {build / 'main.pdf'}", flush=True)


def job_is_running(pid):
    if not pid:
        return False
    result = subprocess.check_output(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"])
    return f'"{pid}"'.encode("ascii") in result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-pid", type=int)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    benchmark = project / "data" / "four_benchmark_eval"
    root = benchmark / f"full_results-local-{READER}"
    paper = project / "paper" / "elsevier_evidence_rag"
    manifest = json.loads((benchmark / "dataset_manifest.json").read_bytes())
    expected = {
        "DeepCAD": manifest["DeepCAD"]["official_split"]["test"],
        "TriView2CAD": manifest["TriView2CAD"]["synthetic_metadata_rows"],
        "FCM": manifest["FCM"]["machining_image_label_pairs"],
    }
    first_pass = True
    while True:
        scores = completed_scores(root, expected)
        build = project.parent / "tmp" / "pdfs" / "text-full-build"
        changed = publish_rows(paper, scores)
        if changed or first_pass:
            rebuild_pdf(paper, build)
            print("Published", ", ".join(d for d in DATASETS if (d, METHODS[0]) in scores), flush=True)
            first_pass = False
        if len(scores) == len(METHODS)*len(DATASETS) or not job_is_running(args.job_pid):
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
