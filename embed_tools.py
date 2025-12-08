import json
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    # 1) Load the JSON
    with open("multimedia_apis.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    ids = [n["id"] for n in nodes]
    descs = [n["desc"] for n in nodes]

    print(f"Loaded {len(ids)} tools")

    # 2) Load a sentence embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # 3) Encode descriptions -> embeddings (shape: N x D)
    emb = model.encode(descs, normalize_embeddings=True)
    print(f"Embeddings shape: {emb.shape}")

    # 4) Save to CSV: id + embedding dimensions
    df = pd.DataFrame(emb)
    df.insert(0, "id", ids)
    df.to_csv("tool_embeddings.csv", index=False)
    print("Wrote tool_embeddings.csv")

if __name__ == "__main__":
    main()
