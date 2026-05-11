import json, pandas as pd
from pathlib import Path
from typing import Dict, List, Set
from .documents import Document

def load_docs_jsonl(path: Path) -> List[Document]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            docs.append(Document(id=str(o["id"]), text=str(o["text"]),
                                 source=str(o.get("source","")), page=o.get("page")))
    return docs

def load_qrels_csv(path: Path) -> Dict[str, Set[str]]:
    df = pd.read_csv(path)
    df = df[df["label"] > 0]
    out: Dict[str, Set[str]] = {}
    for q, sub in df.groupby("query"):
        out[str(q)] = set(map(str, sub["doc_id"].tolist()))
    return out
