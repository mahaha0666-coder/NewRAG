"""Local mechanical drawing extraction CLI; use python -m scripts.parse_drawing."""
import argparse
from pathlib import Path

from src.tools.drawing_extraction.document_parser import parse_drawing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paddle-python", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".venv-paddle-document/bin/python")
    parser.add_argument("--paddle-endpoint", default="http://127.0.0.1:18022/v1")
    parser.add_argument("--semantic-endpoint")
    parser.add_argument("--semantic-model")
    parser.add_argument("--semantic-max-tokens", type=int, default=1800)
    parser.add_argument("--semantic-timeout", type=int, default=600)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--paddle-max-tokens", type=int, default=4096)
    parser.add_argument("--paddle-max-pass-attempts", type=int, default=1)
    parser.add_argument("--feature-memory", type=Path)
    parser.add_argument("--expected-snapshot")
    args = parser.parse_args()
    result = parse_drawing(args.pdf, args.output, args.paddle_python, args.paddle_endpoint,
                           args.semantic_endpoint, args.semantic_model, args.dpi,
                           args.semantic_max_tokens, args.semantic_timeout,
                           feature_memory=args.feature_memory, expected_snapshot=args.expected_snapshot,
                           paddle_max_tokens=args.paddle_max_tokens,
                           paddle_max_pass_attempts=args.paddle_max_pass_attempts)
    print(f"{args.output / 'drawing.json'}: {result.extraction_status}; {len(result.issues)} issues")
    if any(p.paddle_status == "failed" or p.semantic_status == "failed" for p in result.pages):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
