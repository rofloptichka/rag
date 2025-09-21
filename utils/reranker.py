import json
from typing import Any, Dict, List, Optional
import requests


def rerank_with_openai(
    client: Any,
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 5,
    enabled: bool = False,
    model: str = "gpt-4.1-nano",
    timeout: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using an OpenAI model for relevance scoring.

    - client: OpenAI client instance.
    - query: user query string.
    - documents: list of items with keys like 'text', 'title', 'filename', 'score'.
    - top_k: number of results to return.
    - enabled: if False or docs <= 1, returns the first top_k documents unchanged.
    - model: OpenAI chat model to use.
    - timeout: optional request timeout in seconds.
    """
    if not enabled or len(documents) <= 1:
        return documents[:top_k]

    try:
        # Prepare compact documents list for the prompt
        doc_list = []
        for i, doc in enumerate(documents):
            text_preview = (doc.get("text") or "")[:300]
            title = doc.get("title") or "Без названия"
            filename = doc.get("filename") or ""
            doc_summary = f"[{i}] Раздел: {title}\nФайл: {filename}\nТекст: {text_preview}..."
            doc_list.append(doc_summary)

        docs_text = "\n\n".join(doc_list)
        prompt = (
            "Вы - эксперт по поиску в медицинской базе знаний. Оцените релевантность каждого документа к запросу пользователя.\n\n"
            f"ЗАПРОС: \"{query}\"\n\n"
            "ДОКУМЕНТЫ:\n"
            f"{docs_text}\n\n"
            f"Верните JSON со списком индексов документов, отсортированных по релевантности (от самого релевантного к менее релевантному). Включите только {min(top_k, len(documents))} самых релевантных документов.\n\n"
            "Формат ответа:\n"
            "{\"rankings\": [\n"
            "    {\"index\": 0, \"relevance_score\": 0.95, \"reason\": \"краткое объяснение\"},\n"
            "    {\"index\": 2, \"relevance_score\": 0.87, \"reason\": \"краткое объяснение\"}\n"
            "]}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Вы эксперт по медицинской информации. Оцените релевантность документов точно и объективно."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=timeout,
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        reranked_docs: List[Dict[str, Any]] = []
        for ranking in result.get("rankings", []):
            doc_idx = ranking.get("index")
            relevance_score = ranking.get("relevance_score", 0.0)
            reason = ranking.get("reason", "")
            if isinstance(doc_idx, int) and 0 <= doc_idx < len(documents):
                reranked_doc = documents[doc_idx].copy()
                reranked_doc["rerank_score"] = relevance_score
                reranked_doc["rerank_reason"] = reason
                reranked_doc["vector_score"] = reranked_doc.get("score", 0.0)
                reranked_doc["score"] = relevance_score
                reranked_docs.append(reranked_doc)

        return reranked_docs[:top_k] if reranked_docs else documents[:top_k]

    except Exception:
        # Fail open to vector results
        return documents[:top_k]


def rerank_with_jina(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 5,
    api_key: Optional[str] = None,
    model: str = "jina-reranker-v2-base-multilingual",
    endpoint: str = "https://api.jina.ai/v1/rerank",
    timeout: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Jina AI Reranker (v2 models).

    - query: user query string
    - documents: list of items with keys like 'text', 'title', 'filename', 'score'
    - top_k: number of results to return
    - api_key: JINA_API_KEY
    - model: e.g., 'jina-reranker-v2-base-multilingual'
    - endpoint: Jina rerank endpoint (default v1 path)
    """
    if len(documents) <= 1 or not api_key:
        return documents[:top_k]

    # Prepare documents as strings; include titles to improve relevance
    doc_texts: List[str] = []
    for doc in documents:
        title = doc.get("title")
        text = doc.get("text") or ""
        if title:
            doc_texts.append(f"Title: {title}\n\n{text}")
        else:
            doc_texts.append(text)

    payload = {
        "model": model,
        "query": query,
        "documents": doc_texts,
        "top_n": min(top_k, len(documents)),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json() or {}
        ranked = data.get("data", [])
        reranked_docs: List[Dict[str, Any]] = []
        for item in ranked:
            idx = item.get("index")
            score = item.get("relevance_score")
            if score is None:
                score = item.get("score", 0.0)
            if isinstance(idx, int) and 0 <= idx < len(documents):
                d = documents[idx].copy()
                d["rerank_score"] = float(score) if score is not None else 0.0
                d["vector_score"] = d.get("score", 0.0)
                d["score"] = d["rerank_score"]
                reranked_docs.append(d)
        return reranked_docs[:top_k] if reranked_docs else documents[:top_k]
    except Exception:
        return documents[:top_k]


