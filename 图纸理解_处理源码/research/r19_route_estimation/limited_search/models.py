"""Fold-local text ranking and route factorization baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from research.r19_route_estimation.limited_search.text_views import input_digest, text_views

__all__ = ["TextFeatures", "fit", "predict", "margin_scores"]

EMPTY = "__EMPTY__"


@dataclass
class TextFeatures:
    vectorizers: list
    weights: list[float]
    embedding_path: str | None
    embedding_weight: float

    @classmethod
    def fit(cls, rows: list[dict], parameters: dict) -> TextFeatures:
        texts = [text_views(row) for row in rows]
        vectorizers = []
        for index in range(3):
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3) if index == 0 else (2, 4),
                sublinear_tf=True, max_features=12000, min_df=1)
            vectorizer.fit([views[index] or "empty" for views in texts])
            vectorizers.append(vectorizer)
        return cls(vectorizers, parameters.get("text_weights", [1.0, 0.5, 0.5]),
                   parameters.get("embedding_path"), float(parameters.get("embedding_weight", 1.0)))

    def transform(self, rows: list[dict]) -> csr_matrix:
        texts = [text_views(row) for row in rows]
        blocks = [weight * vectorizer.transform([views[index] or "empty" for views in texts])
                  for index, (vectorizer, weight) in enumerate(zip(self.vectorizers, self.weights, strict=True))]
        if self.embedding_path:
            with np.load(self.embedding_path, allow_pickle=False) as data:
                codes = {str(code): index for index, code in enumerate(data["codes"])}
                shape = data["vectors"].shape[1:]
                vectors = []
                for row in rows:
                    supplied = row.get("_text_vectors")
                    if supplied is not None:
                        vector = np.asarray(supplied["vectors"], dtype=np.float32)
                        expected = supplied["input_sha256"]
                    else:
                        if row["item_code"] not in codes:
                            raise ValueError("Missing pretrained features; use the packaged inference encoder")
                        index = codes[row["item_code"]]
                        vector, expected = data["vectors"][index], str(data["input_hashes"][index])
                    if expected != input_digest(row):
                        raise ValueError("Pretrained features do not match the current frozen input")
                    if vector.shape != shape or not np.isfinite(vector).all():
                        raise ValueError("Invalid pretrained feature dimensions or values")
                    vectors.append(vector)
                vectors = np.stack(vectors)
            for view, weight in enumerate(self.weights):
                blocks.append(csr_matrix(self.embedding_weight * weight * vectors[:, view]))
        return normalize(hstack(blocks, format="csr"))


def fit_head(matrix, labels: list[str], c: float):
    if len(set(labels)) == 1:
        return DummyClassifier(strategy="most_frequent").fit(matrix, labels)
    return LinearSVC(C=c, max_iter=10000, random_state=20260926).fit(matrix, labels)


def margin_scores(head, matrix) -> np.ndarray:
    if len(head.classes_) == 1:
        return np.zeros((matrix.shape[0], 1))
    values = head.decision_function(matrix)
    return np.column_stack([-values, values]) if values.ndim == 1 else values


def signature(route: tuple[str, ...]) -> str:
    return ">".join(route) if route else EMPTY


def fit(rows: list[dict], labels: dict[str, list[str]], parameters: dict) -> dict:
    if set(labels) != {row["item_code"] for row in rows}:
        raise ValueError("Training labels must match training items exactly")
    features = TextFeatures.fit(rows, parameters)
    matrix = features.transform(rows)
    routes = [tuple(labels[row["item_code"]]) for row in rows]
    targets = {
        "complete": [signature(route) for route in routes],
        "body": [signature(route[:-1]) for route in routes],
        "first": [route[0] for route in routes],
        "middle": [signature(route[1:-1]) for route in routes],
        "terminal": [route[-1] for route in routes],
    }
    c = float(parameters.get("c", 3.0))
    heads = {name: fit_head(matrix, values, c) for name, values in targets.items()}
    return {"features": features, "heads": heads, "parameters": parameters,
            "training_item_codes": [row["item_code"] for row in rows], "known_routes": set(routes),
            "training_groups": sorted({row["group_id"] for row in rows}),
            "operations": sorted({op for route in routes for op in route})}


def _tokens(value: str) -> tuple[str, ...]:
    return () if value == EMPTY else tuple(value.split(">"))


def predict(model: dict, rows: list[dict], *, limit: int = 3, allow_training_inputs: bool = False) -> list[dict]:
    if not allow_training_inputs and set(model["training_item_codes"]) & {row["item_code"] for row in rows}:
        raise ValueError("Prediction inputs overlap training items")
    matrix = model["features"].transform(rows)
    heads = model["heads"]
    scores = {name: margin_scores(head, matrix) for name, head in heads.items()}
    parameters = model["parameters"]
    method = parameters.get("method", "segmented")
    result = []
    for index, row in enumerate(rows):
        values = {name: dict(zip(map(str, heads[name].classes_), score[index], strict=True)) for name, score in scores.items()}
        pool = {}
        if method == "complete":
            pool = {_tokens(route): value for route, value in values["complete"].items()}
        elif method == "segmented":
            for body, body_score in values["body"].items():
                for terminal, terminal_score in values["terminal"].items():
                    route = (*_tokens(body), terminal)
                    pool[route] = body_score + float(parameters.get("terminal_weight", 0.5)) * terminal_score
        elif method == "factorized":
            starts = sorted(values["first"].items(), key=lambda pair: -pair[1])[:8]
            middles = sorted(values["middle"].items(), key=lambda pair: -pair[1])[:30]
            for first, first_score in starts:
                for middle, middle_score in middles:
                    for terminal, terminal_score in values["terminal"].items():
                        route = (first, *_tokens(middle), terminal)
                        pool[route] = first_score + middle_score + terminal_score * float(parameters.get("terminal_weight", 0.5))
        else:
            raise ValueError(f"Unknown strategy: {method}")
        bonus = float(parameters.get("known_weight", 0.0))
        if bonus:
            lowest = min(values["complete"].values())
            pool = {route: score + bonus * values["complete"].get(signature(route), lowest) for route, score in pool.items()}
        ordered = sorted(pool.items(), key=lambda pair: (-pair[1], pair[0]))
        result.append({"item_code": row["item_code"], "candidates": [
            {"route": list(route), "score": float(score), "origin": method,
             "complete_route_seen_in_train": route in model["known_routes"], "hard_drawing_evidence": False}
            for route, score in ordered[:limit]]})
    return result
