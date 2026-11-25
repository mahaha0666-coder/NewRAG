"""Calibrate frozen visual and text channels within physical training folds."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

from research.r19_route_estimation.limited_search.models import TextFeatures, fit_head, predict, signature
from research.r19_route_estimation.limited_search.text_views import input_digest


@dataclass
class VisualFeatures:
    text: TextFeatures
    path: str
    mean: np.ndarray
    weight: float

    def visual(self, rows: list[dict]) -> np.ndarray:
        with np.load(self.path, allow_pickle=False) as data:
            lookup = {str(code): index for index, code in enumerate(data["codes"])}
            values = []
            shape = data["vectors"].shape[1:]
            for row in rows:
                supplied = row.get("_visual_vectors")
                if supplied is not None:
                    vector, expected = np.asarray(supplied["vectors"], dtype=np.float32), supplied["input_sha256"]
                else:
                    if row["item_code"] not in lookup:
                        raise ValueError("Missing visual features; use the packaged inference encoder")
                    index = lookup[row["item_code"]]
                    vector, expected = data["vectors"][index], str(data["input_hashes"][index])
                if expected != input_digest(row):
                    raise ValueError("Visual input does not match frozen drawing input")
                if vector.shape != shape or not np.isfinite(vector).all():
                    raise ValueError("Invalid visual feature dimensions or values")
                values.append(vector)
            return np.stack(values)

    @classmethod
    def fit(cls, rows: list[dict], parameters: dict) -> VisualFeatures:
        result = cls(TextFeatures.fit(rows, parameters), parameters["visual_path"], np.zeros((1, 1)),
                     float(parameters.get("visual_weight", 0.5)))
        result.mean = result.visual(rows).mean(axis=0)
        return result

    def transform(self, rows: list[dict]):
        visual = self.visual(rows) - self.mean
        blocks = [self.text.transform(rows)]
        for index in range(visual.shape[1]):
            blocks.append(csr_matrix(self.weight * normalize(visual[:, index])))
        return normalize(hstack(blocks, format="csr"))


def fit_visual(rows: list[dict], labels: dict, parameters: dict) -> dict:
    if set(labels) != {row["item_code"] for row in rows}:
        raise ValueError("Training labels must match training inputs")
    features = VisualFeatures.fit(rows, parameters)
    matrix = features.transform(rows)
    routes = [tuple(labels[row["item_code"]]) for row in rows]
    targets = {"complete": [signature(route) for route in routes], "body": [signature(route[:-1]) for route in routes],
               "terminal": [route[-1] for route in routes]}
    heads = {name: fit_head(matrix, values, float(parameters.get("c", 3))) for name, values in targets.items()}
    return {"features": features, "heads": heads, "parameters": {**parameters, "method": parameters.get("route_method", "segmented")},
            "training_item_codes": list(labels), "known_routes": set(routes),
            "training_groups": sorted({row["group_id"] for row in rows}), "operations": sorted({op for route in routes for op in route})}


def fit_selected(rows: list[dict], labels: dict, parameters: dict) -> dict:
    options = [{**parameters, "visual_weight": weight, "c": c, "route_method": method}
               for weight in (0.25, 0.75) for c in (1, 5) for method in ("complete", "segmented")]
    totals = [Counter() for _ in options]
    denominators = Counter()
    for training, heldout in GroupKFold(n_splits=3).split(rows, groups=[row["group_id"] for row in rows]):
        train, queries = [rows[index] for index in training], [rows[index] for index in heldout]
        train_labels = {row["item_code"]: labels[row["item_code"]] for row in train}
        support = {tuple(route) for route in train_labels.values()}
        cohorts = ["seen" if tuple(labels[row["item_code"]]) in support else "isolated" for row in queries]
        denominators.update(cohorts)
        for option, hits in zip(options, totals, strict=True):
            predictions = predict(fit_visual(train, train_labels, option), queries)
            for row, prediction, cohort in zip(queries, predictions, cohorts, strict=True):
                target = labels[row["item_code"]]
                candidates = prediction["candidates"]
                hits[cohort] += any(candidate["route"] == target for candidate in candidates)
                hits["top1"] += bool(candidates and candidates[0]["route"] == target)
    selected = max(range(len(options)), key=lambda index: (
        sum(totals[index][key] / count for key, count in denominators.items()),
        totals[index]["seen"] + totals[index]["isolated"], totals[index]["top1"], -index))
    model = fit_visual(rows, labels, options[selected])
    model["selection"] = {"scope": "inner_physical_group_folds_only", "denominators": dict(denominators),
        "options": [{"parameters": option, "hits": dict(hits)} for option, hits in zip(options, totals, strict=True)], "selected_index": selected}
    return model
