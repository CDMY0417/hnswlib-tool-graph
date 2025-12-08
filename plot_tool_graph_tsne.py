import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors


def main():
    # --- load embeddings (tool_embeddings.csv) ---
    emb_df = pd.read_csv("tool_embeddings.csv")  # columns: id, dim0, dim1, ...
    tool_ids = emb_df["id"].tolist()
    dim_cols = [c for c in emb_df.columns if c != "id"]
    X = emb_df[dim_cols].values
    print(f"Loaded {len(tool_ids)} embeddings with dim = {len(dim_cols)}")

    # --- load types from multimedia_apis.json ---
    with open("multimedia_apis.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # build maps: id -> (input_types, output_types)
    in_types = {}
    out_types = {}
    for node in data["nodes"]:
        tid = node["id"]
        in_types[tid] = node["input-type"]
        out_types[tid] = node["output-type"]

    # ensure ordering of tools in emb_df matches types
    input_seqs = []
    output_seqs = []
    for tid in tool_ids:
        if tid not in in_types:
            raise ValueError(f"Tool id {tid} not found in multimedia_apis.json")
        input_seqs.append(in_types[tid])
        output_seqs.append(out_types[tid])

    # --- create a "signature" string for coloring ---
    # e.g. "text→image" or "audio,audio→audio"
    signatures = []
    for ins, outs in zip(input_seqs, output_seqs):
        sig = f"{','.join(ins)}→{','.join(outs)}"
        signatures.append(sig)

    signatures = np.array(signatures)
    unique_sigs = np.unique(signatures)
    n_sigs = len(unique_sigs)
    print(f"Found {n_sigs} distinct type signatures")

    # map signature -> index 0..n_sigs-1
    sig_to_idx = {sig: i for i, sig in enumerate(unique_sigs)}
    sig_idx = np.array([sig_to_idx[s] for s in signatures])

    # build discrete colormap
    base_cmap = plt.get_cmap("tab20")
    color_list = [base_cmap(i % base_cmap.N) for i in range(n_sigs)]
    discrete_cmap = mcolors.ListedColormap(color_list)
    boundaries = np.arange(n_sigs + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, discrete_cmap.N)

    # --- run t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=10,      # smaller dataset → smaller perplexity
        learning_rate=200,
        init="random",
        random_state=42,
        metric="euclidean",
        verbose=1,
    )
    X_2d = tsne.fit_transform(X)
    xs = X_2d[:, 0]
    ys = X_2d[:, 1]

    # --- load edges ---
    edges_df = pd.read_csv("tool_graph_edges.csv")  # columns: src_id,dst_id

    # map id -> index in embedding array
    id_to_idx = {tid: i for i, tid in enumerate(tool_ids)}

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # draw edges (thin lines)
    for row in edges_df.itertuples(index=False):
        src = row.src_id
        dst = row.dst_id
        if src not in id_to_idx or dst not in id_to_idx:
            continue
        i = id_to_idx[src]
        j = id_to_idx[dst]
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                linewidth=0.5, alpha=0.4, zorder=1)

    # draw nodes: triangles colored by type signature
    scatter = ax.scatter(
        xs, ys,
        c=sig_idx,
        s=80,
        cmap=discrete_cmap,
        norm=norm,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        zorder=2,
    )

    # annotate each node with the tool id (optional, can be busy)
    for i, tid in enumerate(tool_ids):
        ax.text(xs[i], ys[i], tid,
                fontsize=7,
                ha="center",
                va="center",
                zorder=3)

    ax.set_title("Tool graph (t-SNE on descriptions)\ncolor = input→output type signature")
    ax.axis("off")
    ax.set_aspect("equal")

    # colorbar with signature labels
    cbar = fig.colorbar(
        scatter,
        ax=ax,
        boundaries=boundaries,
        ticks=np.arange(n_sigs),
        fraction=0.03,
        pad=0.02,
    )
    cbar.set_label("input→output type signature")
    cbar.set_ticklabels(unique_sigs)

    plt.tight_layout()
    out_file = "tool_tsne_graph.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out_file}")


if __name__ == "__main__":
    main()
