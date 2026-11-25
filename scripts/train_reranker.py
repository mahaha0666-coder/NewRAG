"""Train the small evidence reranker on leak-free variant-family splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mechrag.reranker import (  # noqa: E402
    HardNegativeMiner,
    MLPReranker,
    load_profiles_jsonl,
    split_pairs_by_family,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", help="JSONL part profiles with frozen embeddings")
    parser.add_argument("output", help="Output PyTorch checkpoint")
    parser.add_argument(
        "--critical-property",
        action="append",
        dest="critical_properties",
        default=[],
        help="Repeat for hole_count, hole_type, coaxial, etc.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates-per-query", type=int, default=20)
    parser.add_argument("--minimum-clip-similarity", type=float, default=0.0)
    args = parser.parse_args()

    critical_properties = args.critical_properties or [
        "hole_count",
        "hole_type",
        "coaxial",
        "symmetry",
    ]
    profiles = load_profiles_jsonl(args.profiles)
    miner = HardNegativeMiner(
        critical_properties,
        candidates_per_query=args.candidates_per_query,
        minimum_clip_similarity=args.minimum_clip_similarity,
    )
    examples = miner.mine(profiles)
    split = split_pairs_by_family(
        examples, test_ratio=args.test_ratio, seed=args.seed
    )
    if not split.train:
        raise ValueError(
            "The family split produced no training pairs. Add positive and hard-negative "
            "variants within each family, or add more families."
        )

    reranker = MLPReranker(seed=args.seed)
    losses = reranker.fit(split.train, epochs=args.epochs)
    reranker.save(args.output)
    accuracy = None
    if split.test:
        correct = sum(
            int((reranker.score(example.features) >= 0.5) == bool(example.label))
            for example in split.test
        )
        accuracy = correct / len(split.test)
    print(
        json.dumps(
            {
                "profiles": len(profiles),
                "mined_pairs": len(examples),
                "train_pairs": len(split.train),
                "test_pairs": len(split.test),
                "dropped_cross_split_pairs": len(split.dropped_cross_split),
                "train_families": list(split.train_families),
                "test_families": list(split.test_families),
                "final_train_loss": losses[-1],
                "test_accuracy": accuracy,
                "checkpoint": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
