# Hybrid Search Implementation - Summary

## Изменения

### ✅ 1. Удалена LLM-реструктуризация
- Удалена функция `_llm_restructure_document()`
- Удалены переменные окружения: `RAG_LLM_RESTRUCTURE_DEFAULT`, `LLM_SUM_MODEL`
- Удален параметр API `llmRestructure`
- Документы теперь обрабатываются напрямую через Docling → Chunking → Qdrant

### ✅ 2. Добавлен гибридный поиск (Dense + Sparse)

#### Архитектура Qdrant
Коллекции теперь используют **named vectors**:
```python
vectors_config={
    "dense": VectorParams(size=3072, distance=COSINE),  # OpenAI embeddings
}
sparse_vectors_config={
    "sparse": SparseVectorParams()  # BM25-style term frequency
}
```

#### Sparse Vector Generation
Функция `create_sparse_vector(text)`:
- Простая токенизация (lowercase, regex `\b\w+\b`)
- Подсчет частоты термов
- Хеширование термов в индексы (32-bit int)
- Возвращает `qmodels.SparseVector(indices, values)`

#### Reciprocal Rank Fusion (RRF)
Функция `reciprocal_rank_fusion(dense_results, sparse_results, k=60)`:
- Формула: `score = Σ(1 / (k + rank))` для каждого результата
- Объединяет скоры из dense и sparse поисков
- Возвращает отсортированные результаты с `rrf_score`

### ✅ 3. Обновлены эндпоинты

#### `/companies/{companyId}/process-document`
**Изменения:**
- Удален параметр `llmRestructure`
- Каждая точка сохраняется с двумя векторами:
  ```python
  vector={
      "dense": dense_vector,    # OpenAI embedding
      "sparse": sparse_vector   # Term frequency
  }
  ```

#### `/companies/{companyId}/search` (Documents)
**Новые параметры:**
- `useHybrid: bool = True` — включить гибридный поиск

**Логика:**
1. **Dense search**: `query_vector=("dense", dense_vec)`
2. **Sparse search**: `query_vector=("sparse", sparse_vec)`
3. **RRF Fusion**: объединение результатов
4. **Optional Reranking**: OpenAI/Jina reranker поверх RRF

**Ответ:**
```json
{
  "results": [...],
  "reranked": true/false,
  "hybrid": true/false,
  "reranker_model": "gpt-4.1-nano",
  "rerank_time_ms": 234
}
```

#### `/companies/{companyId}/process-sendable-file`
- Добавлены sparse vectors при сохранении

#### `/companies/{companyId}/update-sendable-description`
- Обновляет оба вектора (dense + sparse)

#### `/companies/{companyId}/search-sendable`
**Новые параметры:**
- `useHybrid: bool = True`

**Логика:**
- Аналогична document search
- RRF fusion для sendable файлов

## Производительность

### Dense-only (старый подход)
- ✅ Отличная семантическая точность
- ❌ Слабая работа с точными терминами
- ❌ Не учитывает keyword matching

### Hybrid (новый подход)
- ✅ Семантика + ключевые слова
- ✅ Лучше находит точные термины/акронимы
- ✅ RRF балансирует оба подхода
- ⚠️ Немного медленнее (2 поиска вместо 1)

## Миграция существующих данных

### ⚠️ ВАЖНО: Коллекции нужно пересоздать!

Старые коллекции используют `vectors_config: VectorParams(...)`, новые используют named vectors.

**Шаги миграции:**
1. Экспортировать данные (опционально)
2. Удалить старые коллекции: `DELETE /companyId/fuckdrop`
3. Загрузить документы заново через `/process-document`

**Или:**
```python
# Автоматическая миграция (если нужно)
for company_id in companies:
    old_collection = f"docling_{company_id}"
    qdrant.delete_collection(old_collection)
    # Re-process all documents
```

## Настройка

### Environment Variables
```bash
# Гибридный поиск включен по умолчанию
# Можно отключить через query параметры: useHybrid=false
```

### Отключение гибридного поиска
```bash
# В запросе:
GET /companies/123/search?query=test&useHybrid=false
```

## Тестирование

### 1. Процессинг документа
```bash
curl -X POST "http://localhost:8000/companies/test123/process-document" \
  -F "file=@document.pdf" \
  -F "naturalChunking=true"
```

### 2. Гибридный поиск
```bash
curl "http://localhost:8000/companies/test123/search?query=pricing+model&useHybrid=true&limit=5"
```

### 3. Dense-only (для сравнения)
```bash
curl "http://localhost:8000/companies/test123/search?query=pricing+model&useHybrid=false&limit=5"
```

## Метрики качества

### Проверка на:
- ✅ **Exact term matching**: "ISO-9001" должен находить документы с этим термином
- ✅ **Semantic understanding**: "цена" находит "стоимость", "прайс"
- ✅ **Acronyms**: "API" vs "Application Programming Interface"
- ✅ **Multiword queries**: "договор аренды офиса"

### Ожидаемые улучшения:
- 📈 Precision для keyword queries: +15-25%
- 📈 Recall для комбинированных запросов: +10-20%
- ⏱️ Latency: +50-100ms (acceptable trade-off)

## Следующие шаги (опционально)

1. **Weighted Fusion** вместо RRF:
   ```python
   final_score = 0.6 * dense_score + 0.4 * sparse_score
   ```

2. **Colbert reranking** для максимальной точности

3. **Query expansion** через LLM перед поиском

4. **A/B тестирование** hybrid vs dense-only

---

**Статус:** ✅ Готово к тестированию
**Breaking changes:** ⚠️ Требуется миграция коллекций
**Backward compatibility:** ❌ Нет (нужен re-index)
