import json
import pandas as pd
from sentence_transformers import SentenceTransformer

JSON_FILES = [
    ("multimedia_apis.json", "multimedia"),
    ("daily_apis.json", "daily"),
    ("huggingface_apis.json", "huggingface"),
]

def main():
    rows = []

    # 1) Collect ids + descs across all libraries
    for path, lib_name in JSON_FILES:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data["nodes"]:
            tool_id = node["id"]
            desc = node["desc"]
            rows.append({
                "library": lib_name,
                "id": tool_id,
                "desc": desc,
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} tools across libraries: {df['library'].unique()}")

    # 2) Embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    emb = model.encode(df["desc"].tolist(), normalize_embeddings=True)
    print("Embeddings shape:", emb.shape)

    # 3) Save to CSV: library, id, dimensions
    emb_df = pd.DataFrame(emb)
    emb_df.insert(0, "id", df["id"])
    emb_df.insert(0, "library", df["library"])
    emb_df.to_csv("tool_embeddings_all.csv", index=False)
    print("Wrote tool_embeddings_all.csv")

if __name__ == "__main__":
    main()
