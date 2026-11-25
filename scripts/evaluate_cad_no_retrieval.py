"""Score the direct visual baseline for published CAD benchmark assets."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import io
from itertools import islice
import json
import math
from pathlib import Path
import random
import re
from zipfile import ZipFile

from openai import OpenAI
from PIL import Image, ImageDraw

from mechrag.providers import DEEPSEEK_OPENAI_BASE_URL, require_deepseek_api_key


MODEL = "deepseek-flash"
SEED = 1517
TRIVIEW_SIZE = 100
DEEPCAD_SIZE = 100
TRIVIEW_FIELDS = (
    ("column_count", "rounded_rect_horizontal_count"),
    ("pile_columns", "circle_horizontal_count"),
    ("pile_rows", "circle_vertical_count"),
)
DEEPCAD_FIELDS = {"Line3D": "line_segments", "Circle3D": "circles"}


def load_fcm(root):
    source = root / "FCM" / "VLMs4Design" / "dfm_subtractive"
    labels = json.loads((source / "labels.json").read_bytes())
    original = (source / "prompt.txt").read_text(encoding="utf-8")
    classes = [label.strip() for label in original.split("List of machining features:", 1)[1].split(",")]
    options = "\n".join(f"{i}: {label}" for i, label in enumerate(classes, 1))
    prompt = (
        "Which machining features are present in the pictured stock? Select all applicable labels. "
        "Use only the listed labels and include Stock when you see the stock body. "
        'Return only JSON in the format {"labels": [integer IDs]}.\n' + options
    )
    for part in sorted(labels, key=int):
        expected = {str(classes.index(label) + 1) for label in labels[part]}
        yield {"id": part, "prompt": prompt, "reference": sorted(expected),
               "image": (source / f"{part}.png").read_bytes()}


def load_triview(root, limit=TRIVIEW_SIZE, *, shard=None):
    source = root / "TriView2CAD"
    records = json.loads((source / "metadata.json").read_bytes())
    rows = records if limit is None else random.Random(SEED).sample(records, limit)
    rows = sorted(rows, key=lambda row: row["id"])
    if shard is not None:
        if limit is not None or len(rows) != 200_000:
            raise ValueError("TriView2CAD sharding requires all 200,000 public synthetic samples.")
        boundary = len(rows) // 2
        rows = rows[:boundary] if shard == 1 else rows[boundary:]
    prompt = (
        "Examine the plan view at the bottom left of this drawing. The rounded rectangles are pier "
        "columns and the circles are piles. Count (1) rounded rectangles, (2) the number of distinct "
        "left-to-right circle positions, and (3) the number of distinct top-to-bottom circle positions. "
        'Return only JSON with integer values {"column_count": n, "pile_columns": n, "pile_rows": n}.'
    )
    with ZipFile(source / "img_files.zip") as images:
        for row in rows:
            yield {
                "id": str(row["id"]), "prompt": prompt,
                "reference": [f"{key}={row['params'][field]}" for key, field in TRIVIEW_FIELDS],
                "image": images.read(f"img_files/{row['id']}.png"),
            }


def load_deepcad(root, limit=DEEPCAD_SIZE):
    source = root / "DeepCAD"
    split = json.loads((source / "train_val_test_split.json").read_bytes())["test"]
    random.Random(SEED).shuffle(split)
    prompt = (
        "Count the straight line segments and full circles in this CAD sketch. "
        "Small black dots mark the endpoints of the line segments. "
        'Return only JSON with integer values {"line_segments": n, "circles": n}.'
    )
    yielded = 0
    for part in split:
        document = json.loads((source / "test" / "cad_json" / f"{part}.json").read_bytes())
        curves = {}
        for step in document["sequence"]:
            if step["type"] != "Sketch":
                continue
            cells = document["entities"][step["entity"]]["profiles"].values()
            curves = {
                curve["curve"]: curve
                for cell in cells for loop in cell["loops"] for curve in loop["profile_curves"]
            }
            if curves or limit is not None:
                break
        if limit is not None and (not curves or any(c["type"] not in DEEPCAD_FIELDS for c in curves.values())):
            continue
        if not curves:
            raise ValueError(f"No usable sketch in {part}")
        bounds = []
        primitives = []
        counts = dict.fromkeys(DEEPCAD_FIELDS.values(), 0)
        for curve in curves.values():
            if curve["type"] == "Line3D":
                a, b = ((curve[k]["x"], curve[k]["y"]) for k in ("start_point", "end_point"))
            elif curve["type"] == "Circle3D":
                x, y = (curve["center_point"][axis] for axis in ("x", "y"))
                radius = curve["radius"]
                a, b = (x-radius, y-radius), (x+radius, y+radius)
            elif curve["type"] == "Arc3D":
                x, y = (curve["center_point"][axis] for axis in ("x", "y"))
                ref = curve["reference_vector"]
                ref_angle = math.atan2(ref["y"], ref["x"])
                first, last = curve["start_angle"], curve["end_angle"]
                arc = [
                    (x + curve["radius"] * math.cos(angle), y + curve["radius"] * math.sin(angle))
                    for angle in (ref_angle + first + (last - first) * i / 48 for i in range(49))
                ]
                bounds.extend(arc)
                primitives.append(("Arc3D", arc, None))
                continue
            else:
                raise ValueError(f"Unsupported curve in {part}: {curve['type']}")
            counts[DEEPCAD_FIELDS[curve["type"]]] += 1
            bounds.extend((a, b))
            primitives.append((curve["type"], a, b))
        xs, ys = zip(*bounds)
        span = max(max(xs)-min(xs), max(ys)-min(ys))
        if span <= 0:
            continue
        scale = 640 / span
        cx, cy = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2

        def pixel(point):
            return 384 + (point[0]-cx)*scale, 384 - (point[1]-cy)*scale

        image = Image.new("RGB", (768, 768), "white")
        draw = ImageDraw.Draw(image)
        for kind, a, b in primitives:
            if kind == "Arc3D":
                draw.line([pixel(point) for point in a], fill="black", width=2)
                continue
            (ax, ay), (bx, by) = pixel(a), pixel(b)
            if kind == "Line3D":
                draw.line((ax, ay, bx, by), fill="black", width=2)
                for x, y in ((ax, ay), (bx, by)):
                    draw.ellipse((x-3, y-3, x+3, y+3), fill="black")
            else:
                draw.ellipse((ax, by, bx, ay), outline="black", width=2)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        yield {"id": part, "prompt": prompt, "image": buffer.getvalue(),
               "reference": [f"{key}={count}" for key, count in counts.items()]}
        yielded += 1
        if limit is not None and yielded == limit:
            return
    if yielded != (len(split) if limit is None else limit):
        raise ValueError(f"Unexpected number of renderable sketches: {yielded}")


def image_url(data):
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((1536, 1536))
        content = io.BytesIO()
        im.save(content, format="PNG")
    return "data:image/png;base64," + base64.b64encode(content.getvalue()).decode("ascii")


def add_reader_arguments(parser):
    parser.add_argument("--reader-url", help="OpenAI-compatible local endpoint, including /v1")
    parser.add_argument("--reader-model", help="Model alias on the local endpoint")


def make_reader(args):
    if bool(args.reader_url) != bool(args.reader_model):
        raise ValueError("Use --reader-url and --reader-model together.")
    if args.reader_url:
        return OpenAI(api_key="local", base_url=args.reader_url, max_retries=2, timeout=180)
    return OpenAI(api_key=require_deepseek_api_key(), base_url=DEEPSEEK_OPENAI_BASE_URL,
                  max_retries=3, timeout=90)


def results_root(args, *, pilot=False):
    folder = "full_results" if args.full and not pilot else "results"
    if args.reader_model:
        name = re.sub(r"[^a-zA-Z0-9._-]", "_", args.reader_model).strip(".")
        if not name:
            raise ValueError("Use a nonempty model alias.")
        folder = f"{folder}-local-{name}"
    return args.root / folder


def recognize(client, row, dataset, *, retrieved_text=(), retrieved_images=(), reader_model=None):
    content = [{"type": "image_url", "image_url": {"url": image_url(row["image"])}}]
    if retrieved_text:
        content.append({"type": "text", "text": "Retrieved OCR excerpts from this drawing:\n"
                        + "\n\n".join(retrieved_text)})
    if retrieved_images:
        content.append({"type": "text", "text": "Retrieved close-ups of the target image follow. "
                        "They may overlap. Use the overview to avoid double counting."})
        content.extend({"type": "image_url", "image_url": {"url": image_url(crop)}} for crop in retrieved_images)
    content.append({"type": "text", "text": row["prompt"]})
    options = {"seed": SEED} if reader_model else {"extra_body": {"thinking": {"type": "disabled"}}}
    response = client.chat.completions.create(
        model=reader_model or MODEL, temperature=0, max_tokens=350, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
        **options,
    )
    raw = response.choices[0].message.content or ""
    try:
        output = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            output = json.loads(match.group()) if match else {}
        except json.JSONDecodeError:
            output = {}
    if not isinstance(output, dict):
        output = {}
    if dataset == "FCM":
        labels = output.get("labels")
        prediction = {str(value) for value in labels if str(value).strip()} if isinstance(labels, list) else set()
    else:
        keys = (key for key, _ in TRIVIEW_FIELDS) if dataset == "TriView2CAD" else DEEPCAD_FIELDS.values()
        prediction = {f"{key}={output[key]}" for key in keys if key in output}
    return {"id": row["id"], "reference": row["reference"], "prediction": sorted(prediction),
            "raw_response": raw, "model": response.model,
            "usage": response.usage.model_dump() if response.usage else {}}


def measure(dataset, results, method="No retrieval", reader=MODEL):
    exact = true = predicted = gold = 0
    for row in results:
        p, g = set(row["prediction"]), set(row["reference"])
        exact += p == g
        true += len(p & g)
        predicted += len(p)
        gold += len(g)
    return {
        "dataset": dataset, "method": method, "reader": reader,
        "samples": len(results), "exact": exact, "true_positive": true,
        "predicted_items": predicted, "gold_items": gold,
        "acc": 100 * exact / len(results), "recall": 100 * true / gold,
        "f1": 200 * true / (gold + predicted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/four_benchmark_eval"))
    loaders = {"FCM": load_fcm, "TriView2CAD": load_triview, "DeepCAD": load_deepcad}
    parser.add_argument("--dataset", nargs="+", choices=tuple(loaders), required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--full", action="store_true")
    add_reader_arguments(parser)
    args = parser.parse_args()
    client = make_reader(args)
    result_root = results_root(args)
    destination = result_root / "no_retrieval"
    destination.mkdir(parents=True, exist_ok=True)
    scores = []
    for dataset in args.dataset:
        full = args.full and dataset in {"DeepCAD", "TriView2CAD"}
        items = loaders[dataset](args.root, limit=None) if full else loaders[dataset](args.root)
        output = destination / f"{dataset}.jsonl"
        existing = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if output.exists() else []
        completed = {row["id"]: row for row in existing}
        expected = set()
        pilot_path = results_root(args, pilot=True) / "no_retrieval" / f"{dataset}.jsonl"
        pilot = {
            row["id"]: row for row in (json.loads(line) for line in pilot_path.read_text(encoding="utf-8").splitlines())
        } if args.full and pilot_path.exists() else {}

        def pending():
            for item in items:
                ident = item["id"]
                if ident in expected:
                    raise ValueError(f"Duplicate benchmark ID: {ident}")
                expected.add(ident)
                saved = completed.get(ident) or pilot.get(ident)
                if saved and saved["reference"] != item["reference"]:
                    raise ValueError(f"Reference changed for {dataset} {ident}")
                if ident not in completed:
                    yield item, saved

        def predict(pair):
            item, saved = pair
            return saved or recognize(client, item, dataset, reader_model=args.reader_model)

        with output.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            remaining = pending()
            while (batch := list(islice(remaining, args.workers * 3))):
                for result in pool.map(predict, batch):
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed[result["id"]] = result
                    if len(completed) % (100 if args.full else 10) == 0:
                        print(f"{dataset}: {len(completed)} scored", flush=True)
        assert set(completed) == expected
        summary = measure(dataset, completed.values(), reader=args.reader_model or MODEL)
        print(json.dumps(summary), flush=True)
        scores.append(summary)
    name = "scores-" + "-".join(args.dataset) + ".csv"
    with (destination / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)


if __name__ == "__main__":
    main()
