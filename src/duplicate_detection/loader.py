import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .models import Document


def load_jsonl(path: Path) -> List[Document]:
    documents: List[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            documents.append(
                Document(
                    doc_id=str(payload.get("doc_id")),
                    title=payload.get("title"),
                    text=payload.get("text", ""),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in {"doc_id", "title", "text"}
                    },
                )
            )
    return documents


def load_csv(
    path: Path, text_column: str, id_column: str, title_column: Optional[str] = None
) -> List[Document]:
    frame = pd.read_csv(path)
    documents: List[Document] = []
    for _, row in frame.iterrows():
        doc_id = str(row[id_column])
        title = (
            str(row[title_column])
            if title_column and not pd.isna(row[title_column])
            else None
        )
        text = str(row[text_column]) if not pd.isna(row[text_column]) else ""
        metadata = row.to_dict()
        documents.append(
            Document(doc_id=doc_id, title=title, text=text, metadata=metadata)
        )
    return documents


def load_text_directory(
    directory: Path,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
    limit: Optional[int] = None,
) -> List[Document]:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory {directory} not found")

    files = sorted(directory.glob(pattern), key=lambda p: p.name)
    documents: List[Document] = []

    for idx, file in enumerate(files, start=1):
        text = file.read_text(encoding=encoding)
        title = _derive_title(text)
        documents.append(
            Document(
                doc_id=file.stem,
                title=title,
                text=text,
                metadata={"source_path": str(file)},
            )
        )
        if idx % 100 == 0:
            logging.debug("Loaded %d text files", idx)
        if limit is not None and len(documents) >= limit:
            logging.info("Reached limit of %d files", limit)
            break

    return documents


def _derive_title(text: str, max_length: int = 80) -> Optional[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:max_length]
    return None
