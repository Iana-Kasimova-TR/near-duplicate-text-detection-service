#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duplicate detection API demo")
    parser.add_argument("--doc-id", type=str, help="Identifier to send to the API")
    parser.add_argument("--query-file", type=Path, help="Path to a text file with new content")
    parser.add_argument(
        "--top", type=int, default=3, help="Number of top candidates to display",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("DUPLICATE_API_URL", "http://localhost:8000"),
        help="Base URL of the duplicate detection API",
    )
    return parser.parse_args()


def call_evaluate(api_url: str, doc_id: str, text: str) -> dict:
    response = requests.post(
        f"{api_url}/duplicates/evaluate",
        json={"doc_id": doc_id, "text": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()

    if args.query_file:
        text = args.query_file.read_text(encoding="utf-8")
        doc_id = args.doc_id or args.query_file.stem
    elif args.doc_id:
        print("Provide --query-file with the document text when evaluating via the API.")
        sys.exit(1)
    else:
        print("Provide --query-file with the document text to evaluate.")
        sys.exit(1)

    result = call_evaluate(args.api_url, doc_id, text)

    print(f"Evaluated document: {result['evaluated_document_id']}")
    candidates = result.get('candidates', [])
    if not candidates:
        print("No potential duplicates found.")
        return

    for idx, candidate in enumerate(candidates[: args.top], start=1):
        print("-" * 80)
        print(f"Candidate {idx}: {candidate['doc_id']}")
        print(f"  BM25 score: {candidate['bm25_score']:.3f}")
        print(f"  SimHash similarity: {candidate['simhash_similarity']:.3f}")
        print(f"  Embedding cosine similarity: {candidate['embedding_similarity']:.3f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
