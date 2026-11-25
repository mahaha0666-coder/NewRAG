"""Evaluate both OCR retrieval baselines with one local reader and a resumable cache."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
import sqlite3
import threading
from time import monotonic

from evaluate_cad_no_retrieval import add_reader_arguments, make_reader, measure, recognize, results_root
from evaluate_cad_retrieval import DenseTextRanker, LOADERS, METHODS, ocr_records, read_rows, text_context


TEXT_METHODS = ("ocr-bm25", "ocr-bge-m3")
THREAD = threading.local()


def scan_ocr(item, path):
    if path.exists():
        try:
            return json.loads(path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError):
            path.unlink()
    if not hasattr(THREAD, "ocr"):
        from rapidocr_onnxruntime import RapidOCR

        THREAD.ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    return ocr_records(THREAD.ocr, item, path)


def request_key(item, text, model):
    signature = json.dumps([item["prompt"], text, model], ensure_ascii=False).encode("utf-8")
    return sha256(sha256(item["image"]).digest() + signature).hexdigest()


def read_checkpoint(path):
    if not path.exists():
        return {}
    rows = {}
    with path.open("r+b") as stream:
        while line := stream.readline():
            if not line.endswith(b"\n"):
                stream.truncate(stream.tell() - len(line))
                break
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"Duplicate prediction in {path}: {row['id']}")
            rows[row["id"]] = row
    return rows


def check_triview(args, client, ranker):
    item = next(LOADERS["TriView2CAD"](args.root, limit=None, shard=args.triview_shard))
    ranker.encode([item["prompt"]])
    path = args.root / "results" / "ocr_cache" / "TriView2CAD" / f"{item['id']}.json"
    records = scan_ocr(item, path)
    for method in TEXT_METHODS:
        context = text_context(item, records, method, ranker if method == "ocr-bge-m3" else None)
        result = recognize(client, item, "TriView2CAD", retrieved_text=[chunk["text"] for chunk in context],
                           reader_model=args.reader_model)
        if result["model"] != args.reader_model or not result["prediction"]:
            raise ValueError(f"Reader check returned no usable response for {method}.")
    print(f"OCR, BGE-M3 and the local visual reader passed on {item['id']}.", flush=True)


def evaluate_dataset(args, dataset, client, ranker, cache, result_dir):
    if dataset == "TriView2CAD":
        items = LOADERS[dataset](args.root, limit=None, shard=args.triview_shard)
    elif dataset == "DeepCAD":
        items = LOADERS[dataset](args.root, limit=None)
    else:
        items = LOADERS[dataset](args.root)
    pilot_root = results_root(args, pilot=True)
    files = {method: result_dir / method / f"{dataset}.jsonl" for method in TEXT_METHODS}
    rows = {method: read_checkpoint(path) for method, path in files.items()}
    pilots = {method: read_rows(pilot_root / method / f"{dataset}.jsonl") for method in TEXT_METHODS}
    direct = read_rows(result_dir / "no_retrieval" / f"{dataset}.jsonl")
    direct.update(read_rows(pilot_root / "no_retrieval" / f"{dataset}.jsonl"))
    expected = set()
    started = monotonic()

    def pending():
        for item in items:
            ident = item["id"]
            if ident in expected:
                raise ValueError(f"Duplicate sample {dataset}/{ident}")
            expected.add(ident)
            for method in TEXT_METHODS:
                for source in (rows[method], pilots[method], direct):
                    saved = source.get(ident)
                    if saved and (saved["reference"] != item["reference"] or saved["model"] != args.reader_model):
                        raise ValueError(f"Invalid saved prediction for {dataset}/{ident}")
            if any(ident not in rows[method] for method in TEXT_METHODS):
                yield item

    def store_sample(method, item, evidence, result, key, source, stream):
        row = dict(result, id=item["id"], reference=item["reference"], retrieved=evidence,
                   reused_direct=source == "direct", inference_source=source, request_sha256=key)
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows[method][row["id"]] = row

    remaining = pending()
    for file in files.values():
        file.parent.mkdir(parents=True, exist_ok=True)
    with (files["ocr-bm25"].open("a", encoding="utf-8") as bm25_stream,
          files["ocr-bge-m3"].open("a", encoding="utf-8") as bge_stream,
          ThreadPoolExecutor(max_workers=args.ocr_workers) as ocr_pool,
          ThreadPoolExecutor(max_workers=args.workers) as reader_pool):
        streams = dict(zip(TEXT_METHODS, (bm25_stream, bge_stream)))
        while (batch := list(islice(remaining, args.workers * 4))):
            ocr = {
                item["id"]: ocr_pool.submit(scan_ocr, item,
                    args.root / "results" / "ocr_cache" / dataset / f"{item['id']}.json")
                for item in batch if any(item["id"] not in source for source in pilots.values())
            }
            jobs, entries = {}, []
            for item in batch:
                ident = item["id"]
                records = ocr[ident].result() if ident in ocr else []
                ranked = {
                    method: text_context(item, records, method, ranker if method == "ocr-bge-m3" else None)
                    for method in TEXT_METHODS if ident not in pilots[method] and ident not in rows[method]
                }
                for method in TEXT_METHODS:
                    if ident in rows[method]:
                        continue
                    saved = pilots[method].get(ident)
                    evidence = saved["retrieved"] if saved else ranked[method]
                    text = [chunk["text"] for chunk in evidence]
                    key = request_key(item, text, args.reader_model)
                    cached = cache.execute("SELECT output FROM predictions WHERE hash=?", (key,)).fetchone()
                    if saved:
                        result, source = saved, "pilot"
                    elif not evidence and ident in direct:
                        result, source = direct[ident], "direct"
                    elif cached:
                        result, source = json.loads(cached[0]), "identical_input"
                    else:
                        if key not in jobs:
                            jobs[key] = reader_pool.submit(recognize, client, item, dataset,
                                retrieved_text=text, reader_model=args.reader_model)
                            source = "reader"
                        else:
                            source = "identical_input"
                        result = None
                    entries.append((method, item, evidence, result, key, source))
            for method, item, evidence, result, key, source in entries:
                # Save scoring inputs and the response, excluding the original image.
                if result is None:
                    result = jobs[key].result()
                cached_output = {name: result[name] for name in ("prediction", "raw_response", "model", "usage")}
                cache.execute("INSERT OR IGNORE INTO predictions VALUES (?,?)", (key, json.dumps(cached_output)))
                store_sample(method, item, evidence, result, key, source, streams[method])
            for stream in streams.values():
                stream.flush()
            cache.commit()
            progress = {method: len(values) for method, values in rows.items()}
            print(json.dumps({"dataset": dataset, "complete_rows": progress,
                              "elapsed_seconds": round(monotonic()-started, 1)}), flush=True)

    scores = []
    for method, predictions in rows.items():
        if set(predictions) != expected:
            raise ValueError(f"Coverage check failed for {dataset}/{method}")
        score = measure(dataset, predictions.values(), METHODS[method], args.reader_model)
        score["with_retrieved_context"] = sum(bool(row["retrieved"]) for row in predictions.values())
        scores.append(score)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/four_benchmark_eval"))
    parser.add_argument("--dataset", nargs="+", choices=tuple(LOADERS), default=["FCM", "DeepCAD", "TriView2CAD"])
    parser.add_argument("--triview-shard", type=int, choices=(1, 2),
                        help="Evaluate one half of the sorted public synthetic dataset.")
    parser.add_argument("--check-only", action="store_true", help="Check one drawing without recording a score.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--ocr-workers", type=int, default=4)
    add_reader_arguments(parser)
    args = parser.parse_args()
    if not args.reader_url or not args.reader_model:
        parser.error("The full text run needs an explicitly configured local reader.")
    if args.triview_shard and args.dataset != ["TriView2CAD"]:
        parser.error("A TriView2CAD shard must be run by itself.")
    if args.check_only and not args.triview_shard:
        parser.error("The check requires a TriView2CAD shard.")
    args.full = True
    root = results_root(args)
    if args.triview_shard:
        root = root / f"shard-{args.triview_shard}-of-2"
    root.mkdir(parents=True, exist_ok=True)
    client = make_reader(args)
    ranker = DenseTextRanker(args.root / "models" / "bge-m3")
    if args.check_only:
        check_triview(args, client, ranker)
        return
    with sqlite3.connect(root / "text-reader-cache.sqlite") as cache:
        cache.execute("PRAGMA journal_mode=WAL")
        cache.execute("CREATE TABLE IF NOT EXISTS predictions (hash TEXT PRIMARY KEY, output TEXT NOT NULL)")
        for dataset in args.dataset:
            scores = evaluate_dataset(args, dataset, client, ranker, cache, root)
            path = root / f"text-full-{dataset}.csv"
            staging = path.with_suffix(".tmp")
            with staging.open("w", encoding="utf-8", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=list(scores[0]))
                writer.writeheader()
                writer.writerows(scores)
            staging.replace(path)
            print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
