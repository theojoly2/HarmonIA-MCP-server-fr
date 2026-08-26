from typing import List, Any, Tuple, Dict
from collections import defaultdict
import time
import multiprocessing
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# =====================================================================
# OPTIMISATION CPU & PARALLÉLISME
# =====================================================================
try:
    NUM_CORES = multiprocessing.cpu_count()
    # On laisse respirer le système tout en maximisant l'usage
    OPTIMAL_THREADS = min(24, max(4, NUM_CORES))
except NotImplementedError:
    OPTIMAL_THREADS = 8

os.environ["OMP_NUM_THREADS"] = str(OPTIMAL_THREADS)
os.environ["MKL_NUM_THREADS"] = str(OPTIMAL_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(OPTIMAL_THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    torch.set_num_threads(OPTIMAL_THREADS)
except ImportError:
    pass
# =====================================================================

from qdrant_client.models import (
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    PayloadSchemaType,
)

try:
    from .load_documents import config as cf
except ImportError:
    from load_documents import config as cf


client = cf.client
model = cf.model
reranker = cf.reranker
COLLECTION = cf.COLLECTION

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

CANDIDATE_MULTIPLIER: int = int(getattr(cf, "config", {}).get("search", {}).get("candidate_multiplier", 8) if hasattr(cf, "config") else 8)
MIN_CANDIDATES: int = int(getattr(cf, "config", {}).get("search", {}).get("min_candidates", 80) if hasattr(cf, "config") else 80)
RERANK_POOL_SIZE: int = int(getattr(cf, "config", {}).get("search", {}).get("rerank_pool_size", 24) if hasattr(cf, "config") else 24)
MAX_CHUNKS_PER_DOCUMENT: int = int(getattr(cf, "config", {}).get("search", {}).get("max_chunks_per_document", 3) if hasattr(cf, "config") else 3)
AGGREGATION_TOP_K: int = int(getattr(cf, "config", {}).get("search", {}).get("aggregation_top_k", 3) if hasattr(cf, "config") else 3)
AGGREGATION_MAX_WEIGHT: float = float(getattr(cf, "config", {}).get("search", {}).get("aggregation_max_weight", 0.9) if hasattr(cf, "config") else 0.9)
AGGREGATION_MEAN_WEIGHT: float = float(getattr(cf, "config", {}).get("search", {}).get("aggregation_mean_weight", 0.1) if hasattr(cf, "config") else 0.1)
HYBRID_DENSE_WEIGHT: float = float(getattr(cf, "config", {}).get("search", {}).get("hybrid_dense_weight", 0.7) if hasattr(cf, "config") else 0.7)

# Configuration des résumés : détermine si le reranking hybride sur résumé est autorisé
SUMMARY_CFG = getattr(cf, "config", {}).get("summary", {})
SUMMARY_ENABLED = bool(SUMMARY_CFG.get("enabled", False))

_FILENAME_INDEX_CREATED = False


# =====================================================================
# CACHE & ENCODAGE VECTORIEL
# =====================================================================

def _lexical_to_sparse_vector(lexical_weights: dict[Any, Any]):
    return SparseVector(
        indices=[int(k) for k in lexical_weights.keys()],
        values=[float(v) for v in lexical_weights.values()],
    )


@lru_cache(maxsize=512)
def _cached_encode(query_text: str) -> dict[str, Any]:
    """Encode la question et met en cache pour éviter de recalculer les mêmes requêtes"""
    capabilities = cf.MODEL_CAPABILITIES
    outputs: dict[str, Any] = {}

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        result = model.encode([query_text], return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense_vecs = result.get("dense_vecs", [])
        lexical_weights = result.get("lexical_weights", [])
        if len(dense_vecs) > 0:
            dense = dense_vecs[0]
            outputs["dense"] = dense.tolist() if hasattr(dense, "tolist") else list(dense)
        if len(lexical_weights) > 0:
            outputs["sparse"] = _lexical_to_sparse_vector(lexical_weights[0])
        return outputs

    if capabilities["has_dense"]:
        result = model.encode([query_text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense_vecs = result.get("dense_vecs", []) if isinstance(result, dict) else result
        if len(dense_vecs) > 0:
            dense = dense_vecs[0]
            outputs["dense"] = dense.tolist() if hasattr(dense, "tolist") else list(dense)
        return outputs

    if capabilities["has_sparse"]:
        result = model.encode([query_text], return_dense=False, return_sparse=True, return_colbert_vecs=False)
        lexical_weights = result.get("lexical_weights", [])
        if len(lexical_weights) > 0:
            outputs["sparse"] = _lexical_to_sparse_vector(lexical_weights[0])
        return outputs

    raise ValueError("No supported vector output available from model")


# =====================================================================
# OUTILS DE RERANKING PAR BATCH (THREADS)
# =====================================================================

def _rerank_batch(batch):
    """Fonction atomique exécutée par les threads pour reranker un lot"""
    scores = reranker.compute_score(batch, normalize=True)
    if not isinstance(scores, (list, tuple)):
        scores = scores.tolist() if hasattr(scores, "tolist") else [scores]
    return scores


def parallel_rerank(pairs, batch_size=24):
    """Orchestre le reranking en parallèle sur tous les cœurs disponibles"""
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    all_scores = []

    with ThreadPoolExecutor(max_workers=OPTIMAL_THREADS) as ex:
        results = list(ex.map(_rerank_batch, batches))

    for r in results:
        all_scores.extend(r)
    return all_scores


# =====================================================================
# FONCTIONS CŒUR DE RECHERCHE
# =====================================================================

def _run_chunk_search(search_terms: str, raw_limit: int, tags: list = None):
    query_vectors = _cached_encode(search_terms)
    query_filter = None
    if tags:
        query_filter = Filter(must=[FieldCondition(key="tags", match=MatchAny(any=tags))])

    dense_res, sparse_res = None, None
    if "dense" in query_vectors:
        dense_res = client.query_points(
            collection_name=COLLECTION, query=query_vectors["dense"], query_filter=query_filter,
            using=DENSE_VECTOR_NAME, limit=raw_limit, with_payload=True,
        )
    if "sparse" in query_vectors:
        sparse_res = client.query_points(
            collection_name=COLLECTION, query=query_vectors["sparse"], query_filter=query_filter,
            using=SPARSE_VECTOR_NAME, limit=raw_limit, with_payload=True,
        )

    if dense_res is None and sparse_res is None:
        raise ValueError("No query vectors generated")
    if sparse_res is None:
        return dense_res
    if dense_res is None:
        return sparse_res

    merged: dict[Any, dict[str, Any]] = {}
    for p in dense_res.points:
        merged[p.id] = {"point": p, "dense": float(p.score), "sparse": 0.0}
    for p in sparse_res.points:
        entry = merged.get(p.id)
        if entry is None:
            entry = {"point": p, "dense": 0.0, "sparse": 0.0}
            merged[p.id] = entry
        entry["sparse"] = float(p.score)

    def minmax(scores: dict[Any, float]) -> dict[Any, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        vmin, vmax = min(vals), max(vals)
        if vmax <= vmin:
            return {k: 0.0 for k in scores.keys()}
        rng = vmax - vmin
        return {k: (v - vmin) / rng for k, v in scores.items()}

    dense_norm = minmax({pid: v["dense"] for pid, v in merged.items()})
    sparse_norm = minmax({pid: v["sparse"] for pid, v in merged.items()})

    for pid, v in merged.items():
        hybrid = HYBRID_DENSE_WEIGHT * dense_norm.get(pid, 0.0) + (1.0 - HYBRID_DENSE_WEIGHT) * sparse_norm.get(pid, 0.0)
        v["hybrid"] = hybrid
        try:
            v["point"].score = hybrid
        except Exception:
            pass

    sorted_points = [
        merged[pid]["point"]
        for pid in sorted(merged.keys(), key=lambda pid: merged[pid]["hybrid"], reverse=True)
    ][:raw_limit]

    class QueryResult:
        def __init__(self, points):
            self.points = points

    return QueryResult(sorted_points)


def _limit_chunks_per_document(points: list[Any], max_chunks_per_document: int = MAX_CHUNKS_PER_DOCUMENT) -> list[Any]:
    counts: dict[str, int] = defaultdict(int)
    limited: list[Any] = []
    for point in points:
        payload = point.payload or {}
        document_id = str(payload.get("document_id") or payload.get("filename") or point.id)
        if counts[document_id] >= max_chunks_per_document:
            continue
        limited.append(point)
        counts[document_id] += 1
    return limited


def _compute_document_aggregated_score(scores: list[float], top_k: int = AGGREGATION_TOP_K, max_weight: float = AGGREGATION_MAX_WEIGHT, mean_weight: float = AGGREGATION_MEAN_WEIGHT) -> float:
    if not scores:
        return 0.0
    sorted_scores = sorted(scores, reverse=True)
    best_score = sorted_scores[0]
    top_scores = sorted_scores[:max(1, top_k)]
    mean_top_score = sum(top_scores) / len(top_scores)
    weight_sum = max_weight + mean_weight
    if weight_sum <= 0: return best_score
    return ((max_weight / weight_sum) * best_score) + ((mean_weight / weight_sum) * mean_top_score)


def _group_best_chunk_per_document(points: list[Any]) -> list[dict[str, Any]]:
    grouped: Dict[str, dict[str, Any]] = {}
    for rank, point in enumerate(points):
        payload = point.payload or {}
        score = float(getattr(point, "score", 0.0) or 0.0)
        document_id = str(payload.get("document_id") or payload.get("filename") or f"doc_{rank}")

        current = grouped.get(document_id)
        if current is None:
            grouped[document_id] = {
                "document_id": document_id,
                "filename": str(payload.get("filename") or document_id),
                "scores": [score],
                "best_score": score,
                "best_chunk_index": int(payload.get("chunk_index", 0)),
                "best_chunk_text": str(payload.get("text", "")),
            }
            continue

        current["scores"].append(score)
        if score > current["best_score"]:
            current["best_score"] = score
            current["best_chunk_index"] = int(payload.get("chunk_index", 0))
            current["best_chunk_text"] = str(payload.get("text", ""))

    aggregated_results: list[dict[str, Any]] = []
    for doc in grouped.values():
        doc["aggregated_score"] = _compute_document_aggregated_score(
            scores=doc["scores"], top_k=AGGREGATION_TOP_K, max_weight=AGGREGATION_MAX_WEIGHT, mean_weight=AGGREGATION_MEAN_WEIGHT,
        )
        aggregated_results.append(doc)
    return sorted(aggregated_results, key=lambda x: (x["aggregated_score"], x["best_score"]), reverse=True)


def _resolve_chunk0_metadata_batch(document_ids: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not document_ids:
        return result
    try:
        points, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchAny(any=document_ids)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                ]
            ),
            limit=len(document_ids),
            with_payload=["doc_summary", "document_id"],
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("document_id")
            if doc_id:
                result[doc_id] = {"id": p.id, "doc_summary": payload.get("doc_summary", "")}
    except Exception as e:
        print(f"[!] Erreur récupération métadonnées : {e}")
    return result


# =====================================================================
# PIPELINE PRINCIPAL (CASCADE + ENRICHISSEMENT + PARALLÉLISME)
# =====================================================================

def retrieve_search_documents(
    search_terms: str,
    tags: list = None,
    limit: int = None,
) -> List[Tuple[str, str, str, Any, float, list, str, int]]:
    #                                                  ^^^^ document_id  ^^^^ best_chunk_index
    if limit is None:
        limit = RERANK_POOL_SIZE

    global _FILENAME_INDEX_CREATED
    if not _FILENAME_INDEX_CREATED:
        try:
            client.create_payload_index(collection_name=COLLECTION, field_name="filename", field_schema=PayloadSchemaType.TEXT)
        except Exception:
            pass
        finally:
            _FILENAME_INDEX_CREATED = True

    try:
        t_start = time.time()
        raw_limit = max(800, RERANK_POOL_SIZE * 15, limit * 20)

        results = _run_chunk_search(search_terms, raw_limit, tags=tags)
        points = getattr(results, "points", [])

        if not points:
            return []

        doc_tags_map = {}
        for p in points:
            payload = p.payload or {}
            doc_id = str(payload.get("document_id") or payload.get("filename") or p.id)
            if doc_id not in doc_tags_map:
                t = payload.get("tags", [])
                doc_tags_map[doc_id] = [t] if isinstance(t, str) else list(t)

        points = _limit_chunks_per_document(points, MAX_CHUNKS_PER_DOCUMENT)
        grouped_docs = _group_best_chunk_per_document(points)

        candidate_docs = grouped_docs[:RERANK_POOL_SIZE]

        doc_ids_needed = [doc["document_id"] for doc in candidate_docs]
        chunk0_metadata_map = _resolve_chunk0_metadata_batch(doc_ids_needed)

        for doc in candidate_docs:
            chunk0_info = chunk0_metadata_map.get(doc["document_id"], {})
            doc["chunk0_id"] = chunk0_info.get("id")
            doc["doc_summary"] = chunk0_info.get("doc_summary", "")

        # --- ÉTAPE 1 : RERANKING DES CHUNKS EN PARALLÈLE ---
        chunk_pairs = []
        for doc in candidate_docs:
            tags_str = ", ".join(doc_tags_map.get(doc["document_id"], [])) or "Inconnue"
            enriched_chunk = f"Source : {tags_str}\nFichier : {doc.get('filename', '')}\nContenu : {str(doc.get('best_chunk_text', ''))[:1024]}"
            chunk_pairs.append([search_terms, enriched_chunk])

        if chunk_pairs:
            chunk_scores = parallel_rerank(chunk_pairs, batch_size=24)
            for i, doc in enumerate(candidate_docs):
                doc["rerank_score"] = float(chunk_scores[i]) if i < len(chunk_scores) else 0.0
            candidate_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # --- ÉTAPE 2 : RERANKING HYBRIDE DES RÉSUMÉS EN PARALLÈLE ---
        # N'effectuer le reranking sur résumé que si tous les finalists provenaient
        # d'une indexation avec summary.enabled=true. Flag manquant = False (safe).
        top_n = min(30, len(candidate_docs))
        finalists = candidate_docs[:top_n]
        rest = candidate_docs[top_n:]

        all_summaries_enabled = SUMMARY_ENABLED and all(
            str(doc.get("summary_enabled", "False")).lower() in ("true", "1", "yes")
            for doc in finalists
        )

        if all_summaries_enabled and finalists:
            summary_pairs = []
            for doc in finalists:
                tags_str = ", ".join(doc_tags_map.get(doc["document_id"], [])) or "Inconnue"
                summary = str(doc.get("doc_summary", "")).strip() or str(doc.get("best_chunk_text", ""))
                enriched_summary = f"Source : {tags_str}\nFichier : {doc.get('filename', '')}\nRésumé : {summary[:1024]}"
                summary_pairs.append([search_terms, enriched_summary])

            if summary_pairs:
                summary_scores = parallel_rerank(summary_pairs, batch_size=24)
                for i, doc in enumerate(finalists):
                    if i < len(summary_scores):
                        doc["rerank_score"] = (0.5 * doc["rerank_score"]) + (0.5 * float(summary_scores[i]))
                finalists.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Reconstruction complète + tri global final
        candidate_docs = finalists + rest
        candidate_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        final_results = [
            (
                doc["filename"],                                    # [0]
                doc.get("best_chunk_text", ""),                     # [1]
                doc.get("doc_summary", "") or "Aucun résumé",       # [2]
                doc.get("chunk0_id"),                               # [3]
                float(doc.get("rerank_score", 0.0)),                # [4]
                doc_tags_map.get(doc["document_id"], []),           # [5]
                doc["document_id"],                                 # [6]
                int(doc.get("best_chunk_index", 0)),                # [7]
            )
            for doc in candidate_docs[:limit]
        ]
        print(f"[Search] Opération complète terminée en {time.time()-t_start:.2f}s")
        return final_results

    except Exception as e:
        import traceback
        print(f"\n[!!! ERROR !!!] Crash in retrieve_search_documents: {e}")
        traceback.print_exc()
        return []


if __name__ == "__main__":
    query = input("Query: ").strip()
    results = retrieve_search_documents(query, limit=8)
    for i, (filename, _, _, chunk0_id, score, tags) in enumerate(results, start=1):
        print(f"[{i}] {filename} | chunk_0_id={chunk0_id} | score={score:.4f} | tags={tags}")
