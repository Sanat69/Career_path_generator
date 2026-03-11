from __future__ import annotations
import chromadb
from sentence_transformers import SentenceTransformer
from config import get_settings

settings = get_settings()

# ──────────────────────────────────────────────
# Initialize embedding model (loaded once at startup)
# ──────────────────────────────────────────────

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        print("Embedding model loaded.")
    return _model


# ──────────────────────────────────────────────
# Initialize ChromaDB (persistent local storage)
# ──────────────────────────────────────────────

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        print(f"ChromaDB initialized at: {settings.chroma_persist_dir}")
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{settings.chroma_collection_name}' ready. Doc count: {_collection.count()}")
    return _collection


# ──────────────────────────────────────────────
# Embed text
# ──────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """Embed a single text string into a vector."""
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a batch (much faster than one-by-one)."""
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return embeddings.tolist()


# ──────────────────────────────────────────────
# Add documents to ChromaDB
# ──────────────────────────────────────────────

def add_documents(
    doc_ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> int:
    """
    Embed and store documents in ChromaDB.
    Returns the number of documents added.
    """
    # Clean None values from metadata — ChromaDB only accepts str/int/float/bool
    metadatas = [
        {k: (v if v is not None else "") for k, v in meta.items()}
        for meta in metadatas
    ]

    collection = get_collection()

    # Check for existing docs to avoid duplicates
    existing = set()
    try:
        existing_docs = collection.get(ids=doc_ids)
        if existing_docs and existing_docs["ids"]:
            existing = set(existing_docs["ids"])
    except Exception:
        pass

    # Filter out already-existing docs
    new_ids = []
    new_texts = []
    new_metadatas = []
    for i, doc_id in enumerate(doc_ids):
        if doc_id not in existing:
            new_ids.append(doc_id)
            new_texts.append(texts[i])
            new_metadatas.append(metadatas[i])

    if not new_ids:
        print("No new documents to add (all already exist).")
        return 0

    # Batch embed
    print(f"Embedding {len(new_ids)} new documents...")
    embeddings = embed_texts(new_texts)

    # Upsert into ChromaDB in batches of 100
    batch_size = 100
    for start in range(0, len(new_ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=new_ids[start:end],
            embeddings=embeddings[start:end],
            documents=new_texts[start:end],
            metadatas=new_metadatas[start:end],
        )

    total = collection.count()
    print(f"Added {len(new_ids)} documents. Collection total: {total}")
    return len(new_ids)


def get_doc_count() -> int:
    """Return total documents in the collection."""
    try:
        return get_collection().count()
    except Exception:
        return 0
