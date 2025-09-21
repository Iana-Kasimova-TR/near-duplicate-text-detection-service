import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from duplicate_detection.embedder import Embedder
from duplicate_detection.loader import load_text_directory
from duplicate_detection.models import DetectionConfig, Document
from duplicate_detection.service import DuplicateDetectionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate duplicate-detection submission CSV"
    )
    parser.add_argument(
        "dataset", type=Path, help="Path to directory containing .txt files"
    )
    parser.add_argument("output", type=Path, help="Where to write the submission CSV")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model for embeddings",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of documents loaded (for testing)"
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=200,
        help="Number of BM25 candidates to consider per document",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Maximum number of candidates retained per document after scoring",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()


def build_service(
    documents: List[Document], args: argparse.Namespace
) -> DuplicateDetectionService:
    config = DetectionConfig(
        bm25_top_k=args.bm25_top_k,
        max_candidates=args.max_candidates,
    )
    embedder = Embedder(args.model)
    return DuplicateDetectionService(documents, config=config, embedder=embedder)


def collect_duplicates(
    service: DuplicateDetectionService,
    documents: List[Document],
) -> Dict[str, Set[str]]:
    duplicate_map: Dict[str, Set[str]] = defaultdict(set)

    for index, document in enumerate(documents, start=1):
        logging.debug(
            "Evaluating document %s (%d/%d)", document.doc_id, index, len(documents)
        )
        result = service.evaluate_document(document.doc_id, include_self=False)
        for candidate in result.candidates:
            doc_a = document.doc_id
            doc_b = candidate.doc_b.doc_id
            if doc_a == doc_b:
                continue
            duplicate_map[doc_a].add(doc_b)
            duplicate_map[doc_b].add(doc_a)

        if index % 100 == 0:
            logging.info("Processed %d/%d documents", index, len(documents))

    return duplicate_map


def format_document_list(duplicates: Set[str]) -> str:
    if not duplicates:
        return "-1"
    return " ".join(sorted(duplicates, key=lambda x: (int(x) if x.isdigit() else x)))


def write_submission(
    output_path: Path,
    documents: List[Document],
    duplicate_map: Dict[str, Set[str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Id,DocumentList\n")
        for doc in sorted(
            documents, key=lambda d: (int(d.doc_id) if d.doc_id.isdigit() else d.doc_id)
        ):
            doc_id = doc.doc_id
            doc_list = format_document_list(duplicate_map.get(doc_id, set()))
            handle.write(f"{doc_id},{doc_list}\n")


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(message)s",
    )

    logging.info("Loading documents from %s", args.dataset)
    documents = load_text_directory(args.dataset, limit=args.limit)
    if not documents:
        raise SystemExit("No documents loaded from the dataset directory")

    logging.info("Loaded %d documents. Building service...", len(documents))
    service = build_service(documents, args)

    logging.info("Collecting duplicate predictions...")
    duplicate_map = collect_duplicates(service, documents)

    logging.info("Writing submission to %s", args.output)
    write_submission(args.output, documents, duplicate_map)
    logging.info("Done.")


if __name__ == "__main__":
    main()
