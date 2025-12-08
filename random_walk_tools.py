"""
사용법
같은 디렉토리에 해당 파일과, layered_nodes.csv, layered_edges_all_lib.csv 세팅
11~17번 줄의 하이퍼파라미터 변경
python random_walk_tools.py
"""
import pandas as pd
import random
import csv

SEQUENCE_LENGTH = 5   # number of tools
NUM_SAMPLES = 200
MAX_TRIES_PER_SAMPLE = 50
START_FROM_LIBRARY = False
EDGES_FILE = "layered_edges_all_lib.csv"
NODES_FILE = "layered_nodes.csv"
OUTPUT_CSV = "random_walk_sequences.csv"

def load_graph(nodes_file: str, edges_file: str):
    """Load nodes and edges from CSV and build adjacency + node metadata."""
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    # Node metadata
    node_type = {}   # node_id -> library 또는 tool
    node_library = {}  # node_id -> library name
    node_tool_id = {}  # node_id -> tool id

    for row in nodes_df.itertuples(index=False):
        nid = row.node_id
        node_type[nid] = row.node_type
        node_library[nid] = row.library if isinstance(row.library, str) else ""
        node_tool_id[nid] = row.tool_id if isinstance(row.tool_id, str) else ""

    adjacency = {}
    for row in edges_df.itertuples(index=False):
        src = row.src
        dst = row.dst
        edge_type = row.edge_type
        adjacency.setdefault(src, []).append((dst, edge_type))

    library_nodes = [nid for nid, t in node_type.items() if t == "library"]

    tool_nodes = [nid for nid, t in node_type.items() if t == "tool"]

    if not library_nodes:
        raise ValueError("No library nodes found in layered_nodes.csv")
    if not tool_nodes:
        raise ValueError("No tool nodes found in layered_nodes.csv")

    return adjacency, node_type, node_library, node_tool_id, library_nodes, tool_nodes


def random_walk_from_library(
    adjacency,
    node_type,
    library_nodes,
    seq_len: int
):
    start_lib = random.choice(library_nodes)
    current = start_lib
    tool_sequence = []

    for _ in range(seq_len):
        neighbors = adjacency.get(current, [])
        tool_neighbors = [
            dst for (dst, edge_type) in neighbors
            if node_type.get(dst) == "tool"
        ]
        if not tool_neighbors:
            return None

        next_node = random.choice(tool_neighbors)
        tool_sequence.append(next_node)
        current = next_node

    return "library", start_lib, tool_sequence

def random_walk_from_tool(
    adjacency,
    node_type,
    tool_nodes,
    seq_len: int,
):
    start_tool = random.choice(tool_nodes)
    current = start_tool
    tool_sequence = [start_tool]

    for _ in range(seq_len - 1):
        neighbors = adjacency.get(current, [])
        tool_neighbors = [
            dst for (dst, edge_type) in neighbors
            if node_type.get(dst) == "tool"
        ]
        if not tool_neighbors:
            return None

        next_node = random.choice(tool_neighbors)
        tool_sequence.append(next_node)
        current = next_node

    return "tool", start_tool, tool_sequence


def main():
    adjacency, node_type, node_library, node_tool_id, library_nodes, tool_nodes = load_graph(
        NODES_FILE, EDGES_FILE
    )

    samples = []
    tries = 0
    max_total_tries = NUM_SAMPLES * MAX_TRIES_PER_SAMPLE

    while len(samples) < NUM_SAMPLES and tries < max_total_tries:
        tries += 1

        if START_FROM_LIBRARY:
            walk = random_walk_from_library(
                adjacency=adjacency,
                node_type=node_type,
                library_nodes=library_nodes,
                seq_len=SEQUENCE_LENGTH,
            )
        else:
            walk = random_walk_from_tool(
                adjacency=adjacency,
                node_type=node_type,
                tool_nodes=tool_nodes,
                seq_len=SEQUENCE_LENGTH,
            )

        if walk is None:
            continue

        start_type, start_node, tool_sequence = walk
        samples.append((start_type, start_node, tool_sequence))

    print(
        f"Generated {len(samples)} sequences "
        f"(requested {NUM_SAMPLES}, sequence length {SEQUENCE_LENGTH}, "
        f"START_FROM_LIBRARY={START_FROM_LIBRARY})"
    )

    step_cols = [f"step_{i+1}" for i in range(SEQUENCE_LENGTH)]
    fieldnames = [
        "walk_id",
        "start_type",
        "start_library_node",
        "start_library_name",
        "start_tool_node",
        "start_tool_label",
    ] + step_cols

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (start_type, start_node, tool_nodes_seq) in enumerate(samples, start=1):
            row = {
                "walk_id": idx,
                "start_type": start_type,
                "start_library_node": "",
                "start_library_name": "",
                "start_tool_node": "",
                "start_tool_label": "",
            }

            if start_type == "library":
                row["start_library_node"] = start_node
                row["start_library_name"] = node_library.get(start_node, "")
            else:
                row["start_tool_node"] = start_node
                lib = node_library.get(start_node, "")
                tid = node_tool_id.get(start_node, "")
                row["start_tool_label"] = f"{lib}::{tid}" if lib and tid else start_node

            for step_idx, node_id in enumerate(tool_nodes_seq):
                lib = node_library.get(node_id, "")
                tid = node_tool_id.get(node_id, "")
                label = f"{lib}::{tid}" if lib and tid else node_id
                row[f"step_{step_idx+1}"] = label

            writer.writerow(row)

    print(f"Saved sequences to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()