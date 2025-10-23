import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from openai import OpenAI
from main import (
    create_sparse_vector, 
    embed_texts, 
    company_doc_collection,
    EMBED_DIMS
)

client_openai = OpenAI()
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def migrate_company_collection(company_id: str, batch_size: int = 100):
    """Миграция одной коллекции компании"""
    old_collection = company_doc_collection(company_id)
    new_collection = f"{old_collection}_v2"
    
    print(f"🔄 Migrating {old_collection} → {new_collection}")
    
    # 1) Создаем новую коллекцию с named vectors
    try:
        qdrant.create_collection(
            collection_name=new_collection,
            vectors_config={
                "dense": qmodels.VectorParams(
                    size=EMBED_DIMS,
                    distance=qmodels.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams(
                    index=qmodels.SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"✅ Created {new_collection}")
    except Exception as e:
        print(f"⚠️  Collection exists or error: {e}")
    
    # 2) Читаем все данные из старой коллекции
    migrated_count = 0
    cursor = None
    
    while True:
        # Читаем batch
        batch, cursor = qdrant.scroll(
            collection_name=old_collection,
            limit=batch_size,
            offset=cursor,
            with_payload=True,
            with_vectors=True
        )
        
        if not batch:
            break
        
        # 3) Конвертируем в новый формат
        new_points = []
        for point in batch:
            payload = point.payload or {}
            text = payload.get("text", "")
            
            # Получаем старый dense vector
            if isinstance(point.vector, list):
                dense_vec = point.vector
            elif isinstance(point.vector, dict):
                dense_vec = point.vector.get("dense", point.vector)
            else:
                # Если вектора нет, пропускаем
                print(f"⚠️  Skipping point {point.id}: no vector")
                continue
            
            # Создаем sparse vector
            sparse_vec = create_sparse_vector(text)
            
            # Новая точка с named vectors
            new_points.append(qmodels.PointStruct(
                id=point.id,
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec
                },
                payload=payload
            ))
        
        # 4) Сохраняем batch
        if new_points:
            qdrant.upsert(
                collection_name=new_collection,
                points=new_points,
                wait=True
            )
            migrated_count += len(new_points)
            print(f"📦 Migrated {migrated_count} points...")
        
        if cursor is None:
            break
    
    print(f"✅ Migration complete: {migrated_count} points")
    print(f"🔄 To switch: rename {new_collection} → {old_collection}")
    
    return migrated_count

def migrate_all_companies():
    """Найти все коллекции вида docling_* и мигрировать"""
    collections = qdrant.get_collections().collections
    
    for coll in collections:
        name = coll.name
        if name.startswith("docling_") and not name.endswith("_v2"):
            company_id = name.replace("docling_", "")
            try:
                migrate_company_collection(company_id)
            except Exception as e:
                print(f"❌ Error migrating {company_id}: {e}")

if __name__ == "__main__":
    # Вариант 1: Мигрировать одну компанию
    # migrate_company_collection("test123")
    
    # Вариант 2: Мигрировать все
    migrate_all_companies()