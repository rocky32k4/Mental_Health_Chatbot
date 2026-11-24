"""
vectorstore.py
--------------
Phase 3 : Embeddings + Vector Store

Embeddings : sentence-transformers/all-MiniLM-L6-v2 (local, free, no API key)
Vector DB  : ChromaDB (saved to disk)

Run:
    python src/vectorstore.py
"""

import os
import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ─────────────────────────────────────────────
# PATHS & SETTINGS
# ─────────────────────────────────────────────

CHUNKS_FILE      = Path("data/processed/chunks.json")
VECTORSTORE_PATH = Path("vectorstore/chroma_db")
COLLECTION_NAME  = "mental_health_docs"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
DEFAULT_N_RESULTS = 5
BATCH_SIZE        = 50


# ─────────────────────────────────────────────
# SUPPRESS WARNINGS
# ─────────────────────────────────────────────

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_embedding_function():
    """Local sentence-transformer model. Downloads once (~90MB), works offline after."""
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu"
    )


def get_chroma_client():
    """Persistent ChromaDB client — saves to vectorstore/chroma_db/"""
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTORSTORE_PATH))


def get_collection(client=None, embedding_fn=None):
    """Get or create ChromaDB collection."""
    if client is None:
        client = get_chroma_client()
    if embedding_fn is None:
        embedding_fn = get_embedding_function()

    return client.get_or_create_collection(
        name               = COLLECTION_NAME,
        embedding_function = embedding_fn,
        metadata           = {"hnsw:space": "cosine"}
    )


# ─────────────────────────────────────────────
# BUILD  (run once)
# ─────────────────────────────────────────────

def build_vectorstore(force_rebuild=False):
    """
    Embed all chunks and save to ChromaDB.
    Run ONCE after ingestion.py. Loads from disk every time after.

    Args:
        force_rebuild : True  -> delete existing and rebuild
                        False -> skip if already built (default)
    """
    print("=" * 55)
    print("  PHASE 3 — BUILDING VECTOR STORE")
    print("=" * 55)

    client = get_chroma_client()

    # Always delete old collection if force_rebuild
    # Also delete if embedding function conflict exists
    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        if force_rebuild:
            client.delete_collection(COLLECTION_NAME)
            print("  Deleted old collection.")
        else:
            # Check count without triggering embedding conflict
            try:
                col   = client.get_collection(COLLECTION_NAME)
                count = col.count()
                if count > 0:
                    print(f"\n  Already built — {count} embeddings found.")
                    print("  Skipping. Use force_rebuild=True to rebuild.\n")
                    return
            except Exception:
                # Conflict with old embedding function — rebuild
                print("  Old collection conflict. Rebuilding...")
                client.delete_collection(COLLECTION_NAME)

    embedding_fn = get_embedding_function()
    collection   = get_collection(client, embedding_fn)

    # Load chunks
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"\n  {CHUNKS_FILE} not found."
            "\n  Run first: python src/ingestion.py"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n  Chunks loaded  : {len(chunks)}")
    print(f"  Embedding model: {EMBEDDING_MODEL} (local)")
    print(f"  Saving to      : {VECTORSTORE_PATH}")
    print(f"\n  Embedding on CPU — may take 2-5 mins...")
    print(f"  (Model downloads ~90MB on first run only)\n")

    documents = [c["text"]     for c in chunks]
    ids       = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "source"    : c["source"],
            "char_start": str(c["char_start"]),
            "char_end"  : str(c["char_end"]),
        }
        for c in chunks
    ]

    # Embed in batches
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)

        collection.add(
            documents = documents[i:end],
            ids       = ids[i:end],
            metadatas = metadatas[i:end],
        )

        pct = int((end / total) * 100)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"  [{bar}] {pct}%  ({end}/{total})", end="\r")

    print()
    print(f"\n  Done! {collection.count()} embeddings saved to disk.")
    print("=" * 55)


# ─────────────────────────────────────────────
# RETRIEVE  (called on every chat message)
# ─────────────────────────────────────────────

def retrieve(query: str, n_results: int = DEFAULT_N_RESULTS) -> list[dict]:
    """
    Find the most relevant chunks for a user query.

    Args:
        query     : user's message
        n_results : number of chunks to return (default 5)

    Returns:
        List of dicts — text, source, score, chunk_id
    """
    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "\n  Vector store is empty."
            "\n  Run: python src/vectorstore.py"
        )

    results = collection.query(
        query_texts = [query],
        n_results   = min(n_results, collection.count()),
        include     = ["documents", "metadatas", "distances"],
    )

    return [
        {
            "chunk_id": results["ids"][0][i],
            "text":     results["documents"][0][i],
            "source":   results["metadatas"][0][i]["source"],
            "score":    round(results["distances"][0][i], 4),
        }
        for i in range(len(results["ids"][0]))
    ]


# ─────────────────────────────────────────────
# STATUS CHECK
# ─────────────────────────────────────────────

def get_status() -> dict:
    """Check if vector store is ready. Used by main.py on startup."""
    try:
        client = get_chroma_client()
        names  = [c.name for c in client.list_collections()]

        if COLLECTION_NAME not in names:
            return {"status": "not_built", "count": 0}

        collection = client.get_collection(COLLECTION_NAME)
        count      = collection.count()

        return {
            "status"    : "ready" if count > 0 else "empty",
            "collection": COLLECTION_NAME,
            "count"     : count,
            "model"     : EMBEDDING_MODEL,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────
# RUN DIRECTLY
# ─────────────────────────────────────────────

if __name__ == "__main__":

    build_vectorstore()

    print("\n  Testing retrieval...")
    print("-" * 55)

    test_queries = [
        "how to talk to someone feeling suicidal",
        "motivational interviewing techniques",
        "coping strategies for someone in crisis",
    ]

    for query in test_queries:
        print(f"\n  Query : {query}")
        results = retrieve(query, n_results=2)
        for i, r in enumerate(results, 1):
            print(f"\n    [{i}] Source : {r['source']}")
            print(f"        Score  : {r['score']}  (lower = better)")
            print(f"        Text   : {r['text'][:150]}...")

    print("\n  Phase 3 complete — ready for Phase 4")
