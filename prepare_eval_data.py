import json
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_DATASET_ID = "Jongbin-kr/tool_HNSW"
HF_SPLIT = "train"
OUTPUT_CSV = "eval_queries.csv"


def normalize_task_name(task: str) -> Tuple[str, Optional[str]]:
    s = task.strip()
    if "(" in s and s.endswith(")"):
        main, lib = s[:-1].split("(", 1)
        return main.strip(), lib.strip()
    return s, None


def map_graph_nodes_to_keys(graph_nodes: List[str]) -> Dict[Tuple[Optional[str], str], str]:
    """
    graph_nodes: e.g. ["multimedia::Text Paraphraser", "daily::Weather API", ...]
    Returns: (library, tool_name) -> "library::tool_id"
    """
    mapping: Dict[Tuple[Optional[str], str], str] = {}
    for gn in graph_nodes:
        gn = gn.strip()
        if "::" in gn:
            lib, name = gn.split("::", 1)
            lib = lib.strip()
            name = name.strip()
            mapping[(lib, name)] = gn
            # also allow name-only lookup (if unambiguous in this example)
            if (None, name) not in mapping:
                mapping[(None, name)] = gn
        else:
            name = gn
            if (None, name) not in mapping:
                mapping[(None, name)] = gn
    return mapping


def map_invoking_to_keys(invoking_nodes_json: str,
                         graph_nodes: List[str]) -> Optional[List[str]]:
    """
    Map gold invoking sequence to list of "library::tool_id" keys.
    Returns None if any step cannot be mapped.
    """
    try:
        nodes = json.loads(invoking_nodes_json)
    except Exception:
        return None

    mapping = map_graph_nodes_to_keys(graph_nodes)
    seq_keys: List[str] = []

    for node in nodes:
        task = node.get("task", "")
        name, lib = normalize_task_name(task)

        key = None
        if lib is not None:
            key = mapping.get((lib, name))
        else:
            key = mapping.get((None, name))

        if key is None:
            return None

        seq_keys.append(key)

    return seq_keys


def main():
    print(f"Loading dataset {HF_DATASET_ID} ({HF_SPLIT})...")
    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)

    print(f"Loaded {len(ds)} rows")

    model = SentenceTransformer(MODEL_NAME)

    rows = []
    for i, ex in enumerate(ds):
        if not ex.get("correct", False):
            continue

        user_req = ex.get("user_request", "")
        graph_nodes = ex.get("graph_nodes", [])
        invoking_nodes_json = ex.get("invoking_nodes_json", "")

        gold_keys = map_invoking_to_keys(invoking_nodes_json, graph_nodes)
        if not gold_keys:
            continue

        emb = model.encode(user_req, normalize_embeddings=True)
        emb = emb.astype("float32")

        row = {
            "example_index": i,
            "gold_seq": "||".join(gold_keys),
        }
        for d_idx, val in enumerate(emb):
            row[f"d{d_idx}"] = float(val)

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Prepared {len(df)} examples for evaluation")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
