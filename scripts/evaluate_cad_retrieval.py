"""Run document-local text and pixel retrieval on the saved CAD pilot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from html import escape
import io
from itertools import islice
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from evaluate_cad_no_retrieval import (
    MODEL, add_reader_arguments, load_deepcad, load_fcm, load_triview,
    make_reader, measure, recognize, results_root,
)


LOADERS = {"DeepCAD": load_deepcad, "TriView2CAD": load_triview, "FCM": load_fcm}
METHODS = {
    "ocr-bm25": "OCR + BM25",
    "ocr-bge-m3": "OCR + BGE-M3 (dense)",
    "mwparserfromhell": "mwparserfromhell",
    "trafilatura": "Trafilatura",
    "pixelrag-base": "PixelRAG (base)",
    "pixelrag-lora": "PixelRAG",
}
TILE_NAMES = ("Upper left", "Upper right", "Lower left (bottom left)", "Lower right (bottom right)")
OCR_MIN_CONFIDENCE = 0.80
TOP_K = 2


def question(row):
    return row["prompt"].split("Return only JSON", 1)[0].strip()


def tile_layout(width, height):
    return (
        (0, 0, int(width*.6), int(height*.6)),
        (int(width*.4), 0, width, int(height*.6)),
        (0, int(height*.4), int(width*.6), height),
        (int(width*.4), int(height*.4), width, height),
    )


def tokens(text):
    return re.findall(r"\w+", text.lower())


def ocr_records(engine, item, path):
    if path.exists():
        return json.loads(path.read_bytes())
    with Image.open(io.BytesIO(item["image"])) as image:
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        lines, _ = engine(bgr)
        quadrants = [[] for _ in TILE_NAMES]
        for points, text, confidence in lines or ():
            if confidence < OCR_MIN_CONFIDENCE or not tokens(text):
                continue
            x, y = np.asarray(points).mean(axis=0)
            index = 2 * int(y >= image.height/2) + int(x >= image.width/2)
            quadrants[index].append(text)
    records = [{"region": region, "lines": lines}
               for region, lines in zip(TILE_NAMES, quadrants) if lines]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def text_context(item, records, method, dense_ranker=None):
    chunks = []
    for record in records:
        name, lines = record["region"], record["lines"]
        if method == "mwparserfromhell":
            import mwparserfromhell

            source = f"== {name} ==\n" + "\n".join(lines)
            content = mwparserfromhell.parse(source).strip_code().strip()
        elif method == "trafilatura":
            import trafilatura

            # CAD draws no web markup. Both parsers receive equivalent OCR text.
            paragraphs = "".join(f"<p>{escape(line)}</p>" for line in lines)
            source = f"<html><body><article>{paragraphs}</article></body></html>"
            extracted = trafilatura.extract(source, favor_recall=True, include_comments=False)
            content = f"{name}\n{extracted}" if extracted else ""
        else:
            content = f"{name}\n" + "\n".join(lines)
        if content:
            chunks.append({"region": name, "text": content})
    if not chunks:
        return []
    query = question(item)
    texts = [block["text"] for block in chunks]
    if dense_ranker is not None:
        scores = dense_ranker.score(query, texts)
    else:
        from rank_bm25 import BM25Okapi

        model = BM25Okapi([tokens(value) for value in texts])
        scores = model.get_scores(tokens(query))
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:TOP_K]
    return [dict(chunks[i], score=float(scores[i])) for i in ranked]


class DenseTextRanker:
    """BGE-M3 dense CLS embeddings with cosine similarity."""

    def __init__(self, model_path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(min(torch.get_num_threads(), 6))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True, weights_only=True, add_pooling_layer=False,
        ).eval()
        self.queries = {}

    def encode(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=8192,
                               return_tensors="pt")
        with self.torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state[:, 0]
            return self.torch.nn.functional.normalize(hidden.float(), dim=-1).numpy()

    def score(self, query, texts):
        if query not in self.queries:
            self.queries[query] = self.encode([query])[0]
        scores = self.encode(texts) @ self.queries[query]
        if not np.isfinite(scores).all():
            raise ValueError("BGE-M3 returned an invalid similarity score.")
        return scores


class PixelRetriever:
    """Qwen last-token embeddings and cosine ranking for the official base and adapter."""

    def __init__(self, root, use_adapter):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        torch.set_num_threads(8)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        directory = root / "models"
        base = directory / "Qwen3-VL-Embedding-2B"
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        self.processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.model, info = Qwen3VLForConditionalGeneration.from_pretrained(
            base, dtype=dtype, local_files_only=True, output_loading_info=True,
        )
        missing = [key for key in info["missing_keys"] if key != "lm_head.weight"]
        if missing:
            raise ValueError(f"Embedding model is missing parameters: {missing[:10]}")
        if use_adapter:
            self.merge_adapter(directory / "PixelRAG-LoRA" / "lora_vit" / "ckpt200")
        self.model = self.model.to(self.device).eval()
        if self.device == "cuda" and (torch.backends.cudnn.version() or 0) < 91500:
            patch = self.model.model.visual.patch_embed
            weight = patch.proj.weight
            # Each already-extracted patch produces one spatial Conv3D output.
            # The equivalent linear projection avoids the older cuDNN BF16 fallback.
            def project(pixels):
                projected = pixels.reshape(-1, weight[0].numel()).to(weight.dtype)
                return torch.nn.functional.linear(projected, weight.flatten(1), patch.proj.bias)

            patch.forward = project
        print(f"PixelRAG embedder ready on {self.device}", flush=True)
        self.queries = {}

    def merge_adapter(self, directory):
        from safetensors.torch import load_file

        config = json.loads((directory / "adapter_config.json").read_bytes())
        if (config["peft_type"] != "LORA" or config.get("use_dora") or config.get("use_rslora")
                or config.get("fan_in_fan_out") or config.get("rank_pattern") or config.get("alpha_pattern")
                or config.get("bias", "none") != "none" or config.get("modules_to_save")):
            raise ValueError("Unsupported PixelRAG adapter format.")
        weights = load_file(directory / "adapter_model.safetensors")
        merged = 0
        with self.torch.no_grad():
            for key, left in weights.items():
                if not key.endswith(".lora_A.weight"):
                    continue
                original = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
                # The official adapter uses bare Qwen3VLModel parameter paths.
                name = original if original.startswith("model.") else "model." + original
                target = self.model.get_submodule(name).weight
                right = weights[key.replace(".lora_A.", ".lora_B.")]
                scale = config["lora_alpha"] / left.shape[0]
                target.copy_((target.float() + scale * (right.float() @ left.float())).to(target.dtype))
                merged += 1
        if 2 * merged != len(weights) or merged == 0:
            raise ValueError(f"Expected LoRA A/B matrices, merged {merged} of {len(weights)} weights.")
        print(f"Loaded official PixelRAG adapter into {merged} layers", flush=True)

    def embed(self, images=None, query=None):
        instruction = ("Retrieve images or text relevant to the user's query."
                       if query is not None else "Represent the user's input.")
        contents = ([{"type": "text", "text": query}] if query is not None
                    else [{"type": "image", "image": image} for image in images])
        texts = [
            self.processor.apply_chat_template(
                [{"role": "system", "content": [{"type": "text", "text": instruction}]},
                 {"role": "user", "content": [content]}], tokenize=False, add_generation_prompt=True,
            ) for content in contents
        ]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        batch = batch.to(self.device)
        with self.torch.inference_mode():
            hidden = self.model.model(**batch, use_cache=False).last_hidden_state[:, -1]
            return self.torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().numpy()

    def rank(self, item):
        with Image.open(io.BytesIO(item["image"])) as source:
            source = source.convert("RGB")
            boxes = tile_layout(*source.size)
            tiles = [source.crop(box) for box in boxes]
        reduced = [tile.copy() for tile in tiles]
        for tile in reduced:
            tile.thumbnail((336, 336))
        query = question(item)
        if query not in self.queries:
            self.queries[query] = self.embed(query=query)[0]
        scores = self.embed(images=reduced) @ self.queries[query]
        if not np.isfinite(scores).all():
            raise ValueError(f"Invalid embedding scores for {item['id']}.")
        selection = np.argsort(-scores, kind="stable")[:TOP_K]
        context, images = [], []
        for i in selection:
            output = io.BytesIO()
            tiles[i].save(output, format="PNG")
            images.append(output.getvalue())
            context.append({"region": TILE_NAMES[i], "box": boxes[i], "score": float(scores[i])})
        return context, images


def read_rows(path):
    if not path.exists():
        return {}
    return {row["id"]: row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}


def run(args, method):
    is_pixel = method.startswith("pixelrag-")
    dense_ranker = None
    if is_pixel:
        retriever = PixelRetriever(args.root, use_adapter=method == "pixelrag-lora")
    else:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
        if method == "ocr-bge-m3":
            dense_ranker = DenseTextRanker(args.root / "models" / "bge-m3")
    client = make_reader(args)
    pilot_root = results_root(args, pilot=True)
    result_root = results_root(args)
    destination = result_root / method
    destination.mkdir(parents=True, exist_ok=True)
    scores = []

    for dataset in args.dataset:
        full = args.full and dataset in {"DeepCAD", "TriView2CAD"}
        items = LOADERS[dataset](args.root, limit=None) if full else LOADERS[dataset](args.root)
        direct = read_rows(result_root / "no_retrieval" / f"{dataset}.jsonl")
        pilot = read_rows(pilot_root / method / f"{dataset}.jsonl") if args.full else {}
        output = destination / f"{dataset}.jsonl"
        completed = read_rows(output)
        expected = set()

        def predict(item, evidence, images):
            if not evidence:
                saved = direct.get(item["id"])
                if saved:
                    return dict(saved, retrieved=[], reused_direct=True)
            result = recognize(client, item, dataset,
                               retrieved_text=[chunk["text"] for chunk in evidence] if not is_pixel else (),
                               retrieved_images=images, reader_model=args.reader_model)
            return dict(result, retrieved=evidence, reused_direct=False)

        def pending():
            for item in items:
                ident = item["id"]
                if ident in expected:
                    raise ValueError(f"Duplicate benchmark ID: {ident}")
                expected.add(ident)
                saved = completed.get(ident) or pilot.get(ident) or direct.get(ident)
                if saved and saved["reference"] != item["reference"]:
                    raise ValueError(f"Reference mismatch for {dataset} {ident}.")
                if ident not in completed:
                    yield item

        with output.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Bound in-flight requests and checkpoint before producing the next batch of tiles.
            remaining = pending()
            while (batch := list(islice(remaining, args.workers))):
                jobs = []
                for item in batch:
                    saved = pilot.get(item["id"])
                    if saved:
                        jobs.append(pool.submit(dict, saved))
                        continue
                    if is_pixel:
                        evidence, images = retriever.rank(item)
                    else:
                        path = args.root / "results" / "ocr_cache" / dataset / f"{item['id']}.json"
                        evidence = text_context(item, ocr_records(ocr, item, path), method, dense_ranker)
                        images = ()
                    jobs.append(pool.submit(predict, item, evidence, images))
                for job in jobs:
                    result = job.result()
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    completed[result["id"]] = result
                stream.flush()
                print(f"{METHODS[method]} / {dataset}: {len(completed)} scored", flush=True)
        if set(completed) != expected:
            raise ValueError(f"Unexpected benchmark IDs for {dataset}.")
        score = measure(dataset, completed.values(), METHODS[method], reader=args.reader_model or MODEL)
        score["with_retrieved_context"] = sum(bool(row["retrieved"]) for row in completed.values())
        print(json.dumps(score), flush=True)
        scores.append(score)

    with (destination / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/four_benchmark_eval"))
    parser.add_argument("--dataset", nargs="+", choices=tuple(LOADERS), default=list(LOADERS))
    parser.add_argument("--method", nargs="+", choices=tuple(METHODS), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--full", action="store_true")
    add_reader_arguments(parser)
    arguments = parser.parse_args()
    for selected in arguments.method:
        run(arguments, selected)
