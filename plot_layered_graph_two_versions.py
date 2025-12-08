import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def compute_positions(nodes_df, emb_df):
    # Embedding columns
    dim_cols = [c for c in emb_df.columns if c not in ("library", "id")]
    if not dim_cols:
        raise ValueError("No embedding columns in tool_embeddings_all.csv")

    # Tools only + join with embeddings
    tool_nodes = nodes_df[nodes_df["node_type"] == "tool"].copy()
    merged = tool_nodes.merge(
        emb_df,
        left_on=["library", "tool_id"],
        right_on=["library", "id"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No tool nodes matched embeddings")

    X = merged[dim_cols].values
    tool_node_ids = merged["node_id"].tolist()

    # t-SNE on tool embeddings
    tsne = TSNE(
        n_components=2,
        perplexity=10.0,
        learning_rate=200.0,
        init="random",
        random_state=42,
        metric="euclidean",
        verbose=1,
    )
    X_2d = tsne.fit_transform(X)
    tool_xs = X_2d[:, 0]
    tool_ys = X_2d[:, 1]

    positions = {}
    for nid, x, y in zip(tool_node_ids, tool_xs, tool_ys):
        positions[nid] = (x, y)

    # Library nodes at centroid of their tools
    lib_nodes = nodes_df[nodes_df["node_type"] == "library"].copy()
    for _, row in lib_nodes.iterrows():
        lib_name = row["library"]
        lib_node_id = row["node_id"]
        tools_in_lib = merged[merged["library"] == lib_name]
        if tools_in_lib.empty:
            continue
        xs = X_2d[tools_in_lib.index, 0]
        ys = X_2d[tools_in_lib.index, 1]
        positions[lib_node_id] = (float(xs.mean()), float(ys.mean()))

    # Entry node above centroid
    entry_nodes = nodes_df[nodes_df["node_type"] == "entry"]
    if not entry_nodes.empty:
        entry_id = entry_nodes.iloc[0]["node_id"]
        tool_xs_arr = np.array(tool_xs)
        tool_ys_arr = np.array(tool_ys)
        ex = float(tool_xs_arr.mean())
        ey = float(tool_ys_arr.mean())
        y_span = float(tool_ys_arr.max() - tool_ys_arr.min())
        ey = ey + y_span * 0.2
        positions[entry_id] = (ex, ey)

    return positions


def plot_version(edge_file, positions, nodes_df, out_file, title_suffix):
    edges_df = pd.read_csv(edge_file)

    # Colors by library
    libraries = sorted(
        nodes_df[nodes_df["node_type"].isin(["tool", "library"])]["library"].unique()
    )
    base_cmap = plt.get_cmap("tab10")
    library_to_color = {
        lib: base_cmap(i % base_cmap.N) for i, lib in enumerate(libraries)
    }
    entry_color = "black"

    tool_size = 40.0
    library_size = 120.0
    entry_size = 200.0

    fig, ax = plt.subplots(figsize=(10.0, 8.0))

    # Draw edges
    for _, row in edges_df.iterrows():
        src = row["src"]
        dst = row["dst"]
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="0.7",
            linewidth=0.5,
            alpha=0.6,
            zorder=1,
        )

    # Draw tools and libraries
    for lib in libraries:
        # tools
        tool_subset = nodes_df[
            (nodes_df["node_type"] == "tool") & (nodes_df["library"] == lib)
        ]
        tx, ty = [], []
        for _, row in tool_subset.iterrows():
            nid = row["node_id"]
            if nid not in positions:
                continue
            x, y = positions[nid]
            tx.append(x)
            ty.append(y)

        # library node
        lib_subset = nodes_df[
            (nodes_df["node_type"] == "library") & (nodes_df["library"] == lib)
        ]
        lx, ly = [], []
        for _, row in lib_subset.iterrows():
            nid = row["node_id"]
            if nid not in positions:
                continue
            x, y = positions[nid]
            lx.append(x)
            ly.append(y)

        color = library_to_color[lib]

        if tx:
            ax.scatter(
                tx,
                ty,
                s=tool_size,
                c=[color],
                edgecolors="black",
                linewidths=0.4,
                zorder=2,
            )

        if lx:
            ax.scatter(
                lx,
                ly,
                s=library_size,
                c=[color],
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
                label=lib,
            )

    # Entry node
    entry_nodes = nodes_df[nodes_df["node_type"] == "entry"]
    if not entry_nodes.empty:
        entry_id = entry_nodes.iloc[0]["node_id"]
        if entry_id in positions:
            ex, ey = positions[entry_id]
            ax.scatter(
                [ex],
                [ey],
                s=entry_size,
                c=[entry_color],
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
                label="ENTRY",
            )

    ax.set_title(
        "Layered tool graph (t-SNE on tool embeddings)\n"
        f"{title_suffix}"
    )
    ax.axis("off")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", title="Library / Entry")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def main():
    nodes_df = pd.read_csv("layered_nodes.csv")
    emb_df = pd.read_csv("tool_embeddings_all.csv")

    # Compute positions once (same layout for both versions)
    positions = compute_positions(nodes_df, emb_df)

    # Version 1: same-library-only graph
    plot_version(
        edge_file="layered_edges_same_lib.csv",
        positions=positions,
        nodes_df=nodes_df,
        out_file="layered_graph_tsne_same_lib.png",
        title_suffix="edges only within the same library",
    )

    # Version 2: all-library graph
    plot_version(
        edge_file="layered_edges_all_lib.csv",
        positions=positions,
        nodes_df=nodes_df,
        out_file="layered_graph_tsne_all_lib.png",
        title_suffix="edges allowed across libraries",
    )


if __name__ == "__main__":
    main()
