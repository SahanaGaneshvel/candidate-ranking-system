#!/usr/bin/env python3
"""
Offline precomputation of candidate embeddings.

This script is meant to run with GPU access (e.g., in Colab) and is NOT
part of the timed ranking step. It:
  1. Loads BAAI/bge-small-en-v1.5 via sentence-transformers
  2. Streams candidates.jsonl to build text documents
  3. Embeds in batches, L2-normalizes
  4. Saves cache/candidate_embeddings.npy and cache/candidate_ids.npy
  5. Saves the model to ./models/bge-small-en-v1.5 for offline use
"""

import json
import time
from pathlib import Path
from typing import Generator, Tuple, List

import numpy as np
from sentence_transformers import SentenceTransformer

# Import our document builder
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.document import build_document


def stream_candidates(jsonl_path: Path) -> Generator[Tuple[str, str], None, None]:
    """
    Stream candidates from JSONL file, yielding (candidate_id, document) pairs.

    Streams line-by-line to avoid loading all 100k records into memory.

    Args:
        jsonl_path: Path to candidates.jsonl

    Yields:
        Tuple of (candidate_id, document_text)
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            candidate = json.loads(line)
            candidate_id = candidate.get("candidate_id", candidate.get("id", ""))

            if not candidate_id:
                continue

            document = build_document(candidate)
            yield candidate_id, document


def precompute_embeddings(
    candidates_path: str = "data/candidates.jsonl",
    cache_dir: str = "cache",
    models_dir: str = "models",
    batch_size: int = 256,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> None:
    """
    Precompute and cache embeddings for all candidates.

    Args:
        candidates_path: Path to candidates.jsonl
        cache_dir: Directory to save embedding cache
        models_dir: Directory to save model for offline use
        batch_size: Batch size for embedding (tune for GPU memory)
        model_name: HuggingFace model identifier
    """
    start_time = time.time()

    # Ensure directories exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Save model for offline use
    model_save_path = models_path / "bge-small-en-v1.5"
    print(f"Saving model to: {model_save_path}")
    model.save(str(model_save_path))
    print("Model saved for offline use.")

    # Stream candidates and collect documents
    print(f"Streaming candidates from: {candidates_path}")
    candidates_file = Path(candidates_path)

    if not candidates_file.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    candidate_ids: List[str] = []
    documents: List[str] = []

    for candidate_id, document in stream_candidates(candidates_file):
        candidate_ids.append(candidate_id)
        documents.append(document)

        if len(documents) % 10000 == 0:
            print(f"  Loaded {len(documents):,} candidates...")

    total_candidates = len(candidate_ids)
    print(f"Total candidates loaded: {total_candidates:,}")

    # Embed in batches
    print(f"Embedding documents in batches of {batch_size}...")
    embed_start = time.time()

    # Use encode with batching - sentence-transformers handles this efficiently
    # show_progress_bar=True for visual feedback during long embedding
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize
    )

    embed_time = time.time() - embed_start
    print(f"Embedding completed in {embed_time:.1f}s ({total_candidates / embed_time:.0f} docs/sec)")

    # Ensure float32 and correct shape
    embeddings = embeddings.astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape} (expected: [{total_candidates}, 384])")

    # Verify L2 normalization (norms should be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms - min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")

    # Save embeddings and IDs
    embeddings_path = cache_path / "candidate_embeddings.npy"
    ids_path = cache_path / "candidate_ids.npy"

    print(f"Saving embeddings to: {embeddings_path}")
    np.save(embeddings_path, embeddings)

    print(f"Saving candidate IDs to: {ids_path}")
    np.save(ids_path, np.array(candidate_ids, dtype=object))

    total_time = time.time() - start_time
    print(f"\nPrecomputation complete!")
    print(f"  Total candidates: {total_candidates:,}")
    print(f"  Embeddings file: {embeddings_path} ({embeddings_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  IDs file: {ids_path}")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute candidate embeddings")
    parser.add_argument(
        "--candidates", "-c",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=256,
        help="Batch size for embedding (default: 256)"
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Directory to save embedding cache"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save model for offline use"
    )

    args = parser.parse_args()

    precompute_embeddings(
        candidates_path=args.candidates,
        cache_dir=args.cache_dir,
        models_dir=args.models_dir,
        batch_size=args.batch_size,
    )
