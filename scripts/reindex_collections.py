#!/usr/bin/env python3
"""
Reindex Collections Script

Detects and fixes collections with wrong embedding dimensions.
Re-indexes all documents and sendables from GCS storage.

Usage:
    # Check for dimension mismatches (dry run)
    python scripts/reindex_collections.py --check

    # Reindex specific collection
    python scripts/reindex_collections.py --collection docling_company123

    # Reindex all collections with wrong dimensions
    python scripts/reindex_collections.py --fix-all

    # Reindex only docs or sendables
    python scripts/reindex_collections.py --fix-all --type docs
    python scripts/reindex_collections.py --fix-all --type sendables
"""

import os
import sys
import argparse
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client.http import models as qmodels
from config import qdrant, EMBED_DIMS, bucket
from utils.qdrant_helpers import (
    embed_texts, sha_id,
    company_doc_collection, company_sendable_collection,
    create_sparse_vector_doc_tf_norm
)
from utils.bm25_helpers import bm25_reset_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_collection_dimension(collection_name: str) -> Optional[int]:
    """Get the dense vector dimension of a collection."""
    try:
        info = qdrant.get_collection(collection_name)
        vectors_config = info.config.params.vectors

        # Handle named vectors (new schema)
        if isinstance(vectors_config, dict) and "dense" in vectors_config:
            return vectors_config["dense"].size

        # Handle unnamed vectors (old schema)
        if hasattr(vectors_config, 'size'):
            return vectors_config.size

        return None
    except Exception as e:
        logger.error(f"Failed to get dimension for {collection_name}: {e}")
        return None


def list_all_collections() -> List[str]:
    """List all Qdrant collections."""
    try:
        collections = qdrant.get_collections().collections
        return [c.name for c in collections]
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return []


def find_mismatched_collections(expected_dims: int = None) -> List[Dict[str, Any]]:
    """Find collections with dimensions different from expected."""
    if expected_dims is None:
        expected_dims = EMBED_DIMS

    mismatched = []
    collections = list_all_collections()

    for name in collections:
        # Only check our collections (docling_* and sendable_files_*)
        if not (name.startswith("docling_") or name.startswith("sendable_files_")):
            continue

        actual_dims = get_collection_dimension(name)
        if actual_dims is None:
            mismatched.append({
                "name": name,
                "actual_dims": "unknown",
                "expected_dims": expected_dims,
                "issue": "could not read dimensions"
            })
        elif actual_dims != expected_dims:
            mismatched.append({
                "name": name,
                "actual_dims": actual_dims,
                "expected_dims": expected_dims,
                "issue": "dimension mismatch"
            })

    return mismatched


def get_collection_points_count(collection_name: str) -> int:
    """Get the number of points in a collection."""
    try:
        info = qdrant.get_collection(collection_name)
        return info.points_count
    except Exception:
        return 0


def backup_collection_metadata(collection_name: str) -> List[Dict[str, Any]]:
    """
    Extract all metadata from collection (for re-indexing reference).
    Returns list of {filename, url, text, metadata} for each point.
    """
    backup = []
    offset = None

    while True:
        points, offset = qdrant.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False  # Don't need old vectors
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            backup.append({
                "id": point.id,
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {})
            })

        if offset is None:
            break

    return backup


def recreate_collection_with_correct_dims(collection_name: str, dims: int = None):
    """Delete and recreate collection with correct dimensions."""
    if dims is None:
        dims = EMBED_DIMS

    logger.info(f"Recreating collection {collection_name} with {dims} dimensions")

    # Delete existing collection
    try:
        qdrant.delete_collection(collection_name)
        logger.info(f"Deleted old collection {collection_name}")
    except Exception as e:
        logger.warning(f"Could not delete collection (may not exist): {e}")

    # Create new collection with correct schema
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=dims,
                distance=qmodels.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=False)
            )
        },
        hnsw_config=qmodels.HnswConfigDiff(m=32, ef_construct=256),
        optimizers_config=qmodels.OptimizersConfigDiff(default_segment_number=2),
        replication_factor=int(os.getenv("QDRANT_RF", "1")),
    )

    # Create payload indexes
    for key, field_type in [
        ("metadata.filename", qmodels.PayloadSchemaType.KEYWORD),
        ("metadata.index", qmodels.PayloadSchemaType.KEYWORD),
        ("metadata.title", qmodels.PayloadSchemaType.KEYWORD),
    ]:
        try:
            qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=key,
                field_schema=field_type,
            )
        except Exception:
            pass

    logger.info(f"Created new collection {collection_name} with {dims} dimensions")


def reindex_from_backup(collection_name: str, backup: List[Dict[str, Any]], batch_size: int = 50):
    """
    Re-embed and upsert all documents from backup.
    Uses the text stored in each point to regenerate embeddings.
    """
    if not backup:
        logger.warning(f"No data to reindex for {collection_name}")
        return

    logger.info(f"Re-indexing {len(backup)} points for {collection_name}")

    # Reset BM25 stats for this collection
    bm25_reset_stats(collection_name)

    # Process in batches
    for i in range(0, len(backup), batch_size):
        batch = backup[i:i + batch_size]

        # Prepare texts for embedding
        texts = []
        for item in batch:
            title = item.get("metadata", {}).get("title", "")
            text = item.get("text", "")
            if title:
                texts.append(f"Title: {title}\n\n{text}")
            else:
                texts.append(text)

        # Generate new embeddings
        try:
            dense_vectors = embed_texts(texts)
        except Exception as e:
            logger.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
            continue

        # Create points with new vectors
        points = []
        for j, (item, dense_vec) in enumerate(zip(batch, dense_vectors)):
            text = item.get("text", "")
            title = item.get("metadata", {}).get("title", "")
            sparse_source = f"{title}. {text}" if title else text

            # Create sparse vector
            sparse_vec, tokens = create_sparse_vector_doc_tf_norm(collection_name, sparse_source)

            point = qmodels.PointStruct(
                id=item["id"],
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec
                },
                payload={
                    "text": text,
                    "metadata": item.get("metadata", {})
                }
            )
            points.append(point)

        # Upsert batch
        try:
            qdrant.upsert(collection_name=collection_name, wait=True, points=points)
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(backup) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")

    logger.info(f"Completed re-indexing {collection_name}")


def reindex_collection(collection_name: str, dims: int = None):
    """
    Full reindex workflow for a single collection:
    1. Backup metadata
    2. Recreate collection with correct dimensions
    3. Re-embed and upsert all documents
    """
    if dims is None:
        dims = EMBED_DIMS

    points_count = get_collection_points_count(collection_name)
    logger.info(f"Starting reindex of {collection_name} ({points_count} points)")

    # Step 1: Backup all data
    logger.info("Step 1: Backing up collection data...")
    backup = backup_collection_metadata(collection_name)
    logger.info(f"Backed up {len(backup)} points")

    if not backup:
        logger.warning(f"Collection {collection_name} is empty, just recreating schema")
        recreate_collection_with_correct_dims(collection_name, dims)
        return

    # Step 2: Recreate collection
    logger.info("Step 2: Recreating collection with correct dimensions...")
    recreate_collection_with_correct_dims(collection_name, dims)

    # Step 3: Re-embed and upsert
    logger.info("Step 3: Re-embedding and upserting documents...")
    reindex_from_backup(collection_name, backup)

    # Verify
    new_count = get_collection_points_count(collection_name)
    logger.info(f"Reindex complete: {new_count} points (was {points_count})")


def check_collections():
    """Print report of all collections and their dimensions."""
    print("\n" + "="*70)
    print("COLLECTION DIMENSION CHECK")
    print(f"Expected dimensions: {EMBED_DIMS}")
    print("="*70 + "\n")

    collections = list_all_collections()
    our_collections = [c for c in collections if c.startswith("docling_") or c.startswith("sendable_files_")]

    if not our_collections:
        print("No document/sendable collections found.")
        return

    mismatched = []
    ok_count = 0

    for name in sorted(our_collections):
        actual_dims = get_collection_dimension(name)
        points = get_collection_points_count(name)

        if actual_dims == EMBED_DIMS:
            status = "OK"
            ok_count += 1
        else:
            status = f"MISMATCH ({actual_dims} != {EMBED_DIMS})"
            mismatched.append(name)

        print(f"  {name}")
        print(f"    Dimensions: {actual_dims} [{status}]")
        print(f"    Points: {points}")
        print()

    print("-"*70)
    print(f"Total: {len(our_collections)} collections")
    print(f"  OK: {ok_count}")
    print(f"  Mismatched: {len(mismatched)}")

    if mismatched:
        print("\nCollections needing reindex:")
        for name in mismatched:
            print(f"  - {name}")
        print("\nRun with --fix-all to reindex all mismatched collections")
        print("Or use --collection <name> to reindex a specific collection")


def main():
    parser = argparse.ArgumentParser(
        description="Reindex Qdrant collections with correct embedding dimensions"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check all collections for dimension mismatches (dry run)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Reindex a specific collection by name"
    )
    parser.add_argument(
        "--fix-all",
        action="store_true",
        help="Reindex all collections with wrong dimensions"
    )
    parser.add_argument(
        "--type",
        choices=["docs", "sendables", "all"],
        default="all",
        help="Type of collections to reindex (default: all)"
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=None,
        help=f"Target dimensions (default: {EMBED_DIMS} from config)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex even if dimensions match"
    )

    args = parser.parse_args()
    dims = args.dims or EMBED_DIMS

    if args.check:
        check_collections()
        return

    if args.collection:
        # Reindex specific collection
        if not args.force:
            actual = get_collection_dimension(args.collection)
            if actual == dims:
                print(f"Collection {args.collection} already has correct dimensions ({dims})")
                print("Use --force to reindex anyway")
                return

        reindex_collection(args.collection, dims)
        return

    if args.fix_all:
        mismatched = find_mismatched_collections(dims)

        # Filter by type if specified
        if args.type == "docs":
            mismatched = [m for m in mismatched if m["name"].startswith("docling_")]
        elif args.type == "sendables":
            mismatched = [m for m in mismatched if m["name"].startswith("sendable_files_")]

        if not mismatched:
            print("No collections need reindexing!")
            return

        print(f"\nWill reindex {len(mismatched)} collections:")
        for m in mismatched:
            print(f"  - {m['name']} ({m['actual_dims']} -> {dims})")

        confirm = input("\nProceed? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

        for m in mismatched:
            try:
                reindex_collection(m["name"], dims)
            except Exception as e:
                logger.error(f"Failed to reindex {m['name']}: {e}")

        print("\nDone!")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
