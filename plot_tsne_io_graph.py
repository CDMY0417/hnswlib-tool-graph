import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors


def main():
    # --- Load nodes ---
    nodes = pd.read_csv("nodes.csv")

    # Determine embedding dimension from d0, d1, ...
    dim_cols = [c for c in nodes.columns if c.startswith("d")]
    if not dim_cols:
        raise ValueError("No embedding columns found (expected columns like d0, d1, ...)")

    dim = len(dim_cols)
    print(f"Found embedding dim = {dim}")

    X = nodes[dim_cols].values  # shape: (N, dim)
    node_ids = nodes["id"].values
    input_labels = nodes["input"].values
    output_labels = nodes["output"].values

    # --- Run t-SNE to 2D ---
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="random",
        random_state=42,
        metric="euclidean",
        verbose=1,
    )
    X_2d = tsne.fit_transform(X)  # shape: (N, 2)

    nodes["x"] = X_2d[:, 0]
    nodes["y"] = X_2d[:, 1]

    # Map node id -> row index
    id_to_idx = {int(nid): idx for idx, nid in enumerate(node_ids)}

    # --- Load edges ---
    edges = pd.read_csv("edges.csv")

    # --- Build a discrete color mapping for all labels (input + output) ---
    all_labels = np.unique(np.concatenate([input_labels, output_labels]))
    n_labels = len(all_labels)
    print(f"Number of distinct labels = {n_labels}")

    # Build a discrete colormap with exactly n_labels colors
    base_cmap = plt.get_cmap("tab10")
    color_list = [base_cmap(i % base_cmap.N) for i in range(n_labels)]
    discrete_cmap = mcolors.ListedColormap(color_list)

    # Map label value -> index 0..(n_labels-1)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    input_idx = np.array([label_to_idx[v] for v in input_labels])
    output_idx = np.array([label_to_idx[v] for v in output_labels])

    # Boundaries and norm so each index gets a single solid color
    boundaries = np.arange(n_labels + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, discrete_cmap.N)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw edges first (thin, light)
    for row in edges.itertuples(index=False):
        src = int(row.src)
        dst = int(row.dst)

        if src not in id_to_idx or dst not in id_to_idx:
            continue

        s_idx = id_to_idx[src]
        d_idx = id_to_idx[dst]

        x1 = nodes.at[s_idx, "x"]
        y1 = nodes.at[s_idx, "y"]
        x2 = nodes.at[d_idx, "x"]
        y2 = nodes.at[d_idx, "y"]

        ax.plot([x1, x2], [y1, y2], linewidth=0.2, alpha=0.3)

    # Draw nodes: two overlapping scatters with different markers
    # Left-pointing triangle: color by input label
    scatter_in = ax.scatter(
        nodes["x"],
        nodes["y"],
        c=input_idx,
        s=25,
        marker=">",
        cmap=discrete_cmap,
        norm=norm,
        alpha=0.9,
        edgecolors="none",
        label="input",
    )

    # Right-pointing triangle: color by output label
    scatter_out = ax.scatter(
        nodes["x"],
        nodes["y"],
        c=output_idx,
        s=25,
        marker="<",
        cmap=discrete_cmap,
        norm=norm,
        alpha=0.9,
        edgecolors="none",
        label="output",
    )

    ax.set_title("t-SNE of embeddings\n> input label, < output label")
    ax.axis("off")
    ax.set_aspect("equal")

    # Colorbar for input labels
    cbar_in = fig.colorbar(
        scatter_in,
        ax=ax,
        boundaries=boundaries,
        ticks=np.arange(n_labels),
        fraction=0.046,
        pad=0.04,
    )
    cbar_in.set_label("input label")
    cbar_in.set_ticklabels(all_labels)

    # Colorbar for output labels (same mapping, different label text)
    cbar_out = fig.colorbar(
        scatter_out,
        ax=ax,
        boundaries=boundaries,
        ticks=np.arange(n_labels),
        fraction=0.046,
        pad=0.12,
    )
    cbar_out.set_label("output label")
    cbar_out.set_ticklabels(all_labels)

    plt.tight_layout()

    # --- Save to image file instead of showing ---
    output_file = "tsne_io_graph.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved t-SNE IO graph to {output_file}")


if __name__ == "__main__":
    main()
