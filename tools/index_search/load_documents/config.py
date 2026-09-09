from pathlib import Path
from os import getenv, environ
from typing import Any, List, Tuple, Union
from abc import ABC, abstractmethod
import threading
import time

import requests

from yaml import safe_load
from qdrant_client import QdrantClient
from dotenv import load_dotenv

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType
from huggingface_hub import snapshot_download
from filelock import FileLock

PROJECT_ROOT = Path(__file__).resolve().parent
environ["HF_HOME"] = str(PROJECT_ROOT / ".cache_hf")

# Charge le .env du serveur MCP s'il n'a pas déjà été chargé.
# On utilise override=False pour ne pas écraser les variables déjà présentes
# (notamment celles chargées par server.py avant l'import de ce module).
SERVER_PROJECT_ROOT = PROJECT_ROOT.parents[2]  # Remonte à HarmonIA-MCP-server-fr
server_env_path = SERVER_PROJECT_ROOT / ".env"
if server_env_path.exists():
    load_dotenv(dotenv_path=server_env_path, override=False)
    print(f"✓ Loaded server .env: {server_env_path}")
else:
    load_dotenv(override=False)
    print("✓ Loaded .env from current working directory")

# Vérification debug des variables d'environnement critiques
print(
    f"[ENV DEBUG] SERVER_HOST={getenv('SERVER_HOST')}, "
    f"SERVER_PORT={getenv('SERVER_PORT')}, "
    f"URL_RERANKER_API={getenv('URL_RERANKER_API')}, "
    f"RERANKER_MODEL={getenv('RERANKER_MODEL')}, "
    f"RERANKER_API_KEY_SET={getenv('RERANKER_API_KEY') is not None}"
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = safe_load(f)


# =========================================================
# BASE RERANKER INTERFACE
# =========================================================
class BaseReranker(ABC):
    """Interface commune pour tous les rerankers (local ou API)."""

    @abstractmethod
    def compute_score(self, sentence_pairs, normalize=False, max_length=512):
        """
        sentence_pairs: liste de paires [query, document] ou tuple unique.
        Retourne une liste de scores (ou un score unique si tuple en entrée).
        """
        raise NotImplementedError

    def get_batch_settings(self) -> dict[str, Any]:
        """
        Retourne les paramètres de batch pour ce backend.
        - batch_size: taille du lot
        - parallel: True si le backend supporte le parallélisme
        """
        return {"batch_size": 24, "parallel": True}


# =========================================================
# API EMBEDDING MODEL
# =========================================================
class ApiEmbeddingModel:
    """
    Wrapper pour un modèle d'embedding externe accessible via une API
    compatible OpenAI (/v1/embeddings) ou Albert (/v1/embeddings).
    Retourne des vecteurs denses et, si possible, des poids lexicaux sparse.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 30,
        retry_attempts: int = 1,
        retry_delay: float = 1.0,
    ):
        # On conserve l'URL exacte fournie par l'utilisateur (qui doit contenir
        # le chemin complet, ex: https://albert.api.etalab.gouv.fr/v1/embeddings).
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._url = self.base_url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def encode(self, texts, return_dense=True, return_sparse=False, return_colbert_vecs=False, **kwargs):
        """
        Appelle l'API d'embedding pour encoder une liste de textes.
        Retourne un dict avec 'dense_vecs' et éventuellement 'lexical_weights'.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return {"dense_vecs": [], "lexical_weights": []}

        payload = {
            "input": texts,
            "model": self.model,
            "encoding_format": "float",
        }

        last_exception: Exception | None = None
        attempts = self.retry_attempts + 1

        for attempt in range(attempts):
            try:
                response = requests.post(
                    self._url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                embeddings = sorted(
                    data.get("data", []),
                    key=lambda item: item.get("index", 0),
                )
                dense_vecs = [item.get("embedding") for item in embeddings]

                result: dict[str, Any] = {"dense_vecs": dense_vecs}

                # Certains endpoints (ex: Albert bge-m3) peuvent retourner des sparse weights
                if return_sparse and embeddings and "sparse_weights" in embeddings[0]:
                    result["lexical_weights"] = [
                        item.get("sparse_weights", {}) for item in embeddings
                    ]

                return result

            except requests.exceptions.RequestException as e:
                last_exception = e
                status_code = getattr(e.response, "status_code", None)
                response_body = ""
                try:
                    response_body = e.response.text[:500] if e.response else ""
                except Exception:
                    pass
                if status_code not in (429, 503) and not isinstance(
                    e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
                ):
                    break

                if attempt < self.retry_attempts:
                    print(f"⚠ Embedding API transient error ({status_code}), retrying in {self.retry_delay}s... (attempt {attempt + 1}/{attempts})")
                    time.sleep(self.retry_delay)
                    continue

                break

        raise EmbeddingApiError(f"Embedding API call failed: {last_exception}") from last_exception


class EmbeddingApiError(Exception):
    """Exception levée lors d'un échec d'appel à l'API d'embedding."""
    pass


# =========================================================
# LOCAL RERANKER
# =========================================================
class LocalReranker(BaseReranker):
    """
    GPU  → FP16
    CPU  → ONNX INT8 (Q8 auto export + quantization)
    """

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device == "cuda":
            self.model = self._load_gpu()
        else:
            self.model = self._load_cpu_q8()

    # =====================================================
    # GPU FP16
    # =====================================================
    def _load_gpu(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            dtype=torch.float16,
        )
        return model.to(self.device)

    # =====================================================
    # CPU ONNX + AUTO Q8
    # =====================================================
    def _load_cpu_q8(self):
        base_dir = PROJECT_ROOT / "models/onnx"
        fp32_dir = base_dir / "bge-reranker-v2-m3"
        int8_dir = base_dir / "bge-reranker-v2-m3-int8"

        lock = FileLock(str(int8_dir) + ".lock")

        with lock:

            # -----------------------------
            # 1. si INT8 existe → direct
            # -----------------------------
            if int8_dir.exists() and any(int8_dir.iterdir()):
                print(f"✓ Loading ONNX INT8 reranker: {int8_dir}")
                return ORTModelForSequenceClassification.from_pretrained(
                    str(int8_dir)
                )

            # -----------------------------
            # 2. sinon FP32 ONNX existe ?
            # -----------------------------
            if not fp32_dir.exists() or not any(fp32_dir.iterdir()):
                print("⚠ Export ONNX FP32...")

                snapshot_download(repo_id=self.model_name)

                main_export(
                    model_name_or_path=self.model_name,
                    output=str(fp32_dir),
                    task="text-classification",
                )

                print(f"✓ FP32 ONNX exported → {fp32_dir}")

            # -----------------------------
            # 3. quantization INT8 (Q8)
            # -----------------------------
            print("⚡ Quantizing ONNX → INT8 (Q8)...")

            fp32_model = fp32_dir / "model.onnx"
            int8_model = int8_dir

            int8_model.mkdir(parents=True, exist_ok=True)

            quantize_dynamic(
                model_input=str(fp32_model),
                model_output=str(int8_model / "model.onnx"),
                weight_type=QuantType.QInt8,
            )

            # copier config/tokenizer
            for f in fp32_dir.glob("*"):
                if f.suffix != ".onnx":
                    (int8_model / f.name).write_bytes(f.read_bytes())

            print(f"✓ INT8 ONNX ready → {int8_model}")

        print(f"✓ Loading ONNX INT8 reranker: {int8_dir}")

        return ORTModelForSequenceClassification.from_pretrained(
            str(int8_dir)
        )

    # =====================================================
    # SCORE
    # =====================================================
    def get_batch_settings(self) -> dict[str, Any]:
        return {"batch_size": 24, "parallel": True}

    @torch.inference_mode()
    def compute_score(self, sentence_pairs, normalize=False, max_length=512):
        if isinstance(sentence_pairs, tuple):
            pairs = [sentence_pairs]
        else:
            pairs = list(sentence_pairs)

        queries = [q for q, _ in pairs]
        docs = [d for _, d in pairs]

        inputs = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        logits = self.model(**inputs).logits.view(-1)

        scores = logits.float()

        if normalize:
            scores = torch.sigmoid(scores)

        scores = scores.cpu().tolist()

        if isinstance(sentence_pairs, tuple):
            return scores[0]

        return scores


# =========================================================
# API RERANKER (Albert /v1/rerank)
# =========================================================
class ApiReranker(BaseReranker):
    """
    Reranker via API Albert : https://albert.api.etalab.gouv.fr/v1/rerank
    Format attendu: {query, documents, model, top_n}
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 30,
        retry_attempts: int = 1,
        retry_delay: float = 1.0,
    ):
        # On conserve l'URL exacte fournie par l'utilisateur (qui doit contenir
        # le chemin complet, ex: https://albert.api.etalab.gouv.fr/v1/rerank).
        # On supprime juste un éventuel slash final.
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._url = self.base_url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_batch_settings(self) -> dict[str, Any]:
        return {"batch_size": 100, "parallel": False}

    def compute_score(self, sentence_pairs, normalize=False, max_length=512):
        """
        Albert /v1/rerank attend une query unique + une liste de documents.
        On reconstruit ces champs à partir des paires [query, doc].
        """
        if isinstance(sentence_pairs, tuple):
            pairs = [sentence_pairs]
        else:
            pairs = list(sentence_pairs)

        if not pairs:
            return []

        queries = [q for q, _ in pairs]
        documents = [d for _, d in pairs]

        # Vérification: toutes les queries doivent être identiques
        query = queries[0]
        if any(q != query for q in queries):
            raise ValueError("ApiReranker requires all pairs to share the same query.")

        payload = {
            "query": query,
            "documents": documents,
            "model": self.model,
            "top_n": None,  # Récupérer tous les scores
        }

        last_exception: Exception | None = None
        attempts = self.retry_attempts + 1

        for attempt in range(attempts):
            try:
                response = requests.post(
                    self._url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                scores = [0.0] * len(documents)

                for item in results:
                    idx = item.get("index")
                    score = item.get("relevance_score")
                    if isinstance(idx, int) and 0 <= idx < len(documents) and score is not None:
                        scores[idx] = float(score)

                if isinstance(sentence_pairs, tuple):
                    return scores[0]
                return scores

            except requests.exceptions.RequestException as e:
                last_exception = e
                status_code = getattr(e.response, "status_code", None)
                response_body = ""
                try:
                    response_body = e.response.text[:500] if e.response else ""
                except Exception:
                    pass

                print(f"⚠ Albert rerank API error: status={status_code}, url={self._url}, model={self.model}, api_key_set={self.api_key is not None}, response={response_body}, exception={e}")

                # Retries limités aux erreurs transitoires
                if status_code not in (429, 503) and not isinstance(
                    e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
                ):
                    break

                if attempt < self.retry_attempts:
                    print(f"⚠ Albert rerank API transient error ({status_code}), retrying in {self.retry_delay}s... (attempt {attempt + 1}/{attempts})")
                    time.sleep(self.retry_delay)
                    continue

                break

        # Tous les retries épuisés ou erreur non transitoire
        raise AlbertApiError(f"Albert rerank API call failed: {last_exception}") from last_exception


class AlbertApiError(Exception):
    """Exception levée lors d'un échec d'appel à l'API Albert."""
    pass


# =========================================================
# RERANKER ROUTER (API first, fallback local)
# =========================================================
class RerankerRouter(BaseReranker):
    """
    Tente d'abord l'API Albert.
    En cas d'erreur, fallback immédiat vers LocalReranker.
    Après N erreurs consécutives, l'API est désactivée pour la session.
    """

    def __init__(
        self,
        api_reranker: ApiReranker,
        local_reranker: LocalReranker,
        max_consecutive_errors: int = 3,
    ):
        self.api_reranker = api_reranker
        self.local_reranker = local_reranker
        self.max_consecutive_errors = max(1, max_consecutive_errors)
        self._api_enabled = True
        self._consecutive_errors = 0
        self._lock = threading.Lock()

    def get_batch_settings(self) -> dict[str, Any]:
        return self.api_reranker.get_batch_settings()

    def compute_score(self, sentence_pairs, normalize=False, max_length=512):
        # Si API désactivée, on passe directement en local
        with self._lock:
            api_enabled = self._api_enabled

        if api_enabled:
            try:
                scores = self.api_reranker.compute_score(
                    sentence_pairs, normalize=normalize, max_length=max_length
                )
                with self._lock:
                    self._consecutive_errors = 0
                return scores
            except AlbertApiError as e:
                with self._lock:
                    self._consecutive_errors += 1
                    current = self._consecutive_errors
                    if current >= self.max_consecutive_errors:
                        self._api_enabled = False
                        print(f"🔴 Albert API disabled after {current} consecutive errors. Switching to local reranker permanently.")
                    else:
                        print(f"🟠 Albert API error ({current}/{self.max_consecutive_errors}), falling back to local reranker for this batch.")

                # Fallback local pour ce batch
                return self.local_reranker.compute_score(
                    sentence_pairs, normalize=normalize, max_length=max_length
                )

        # Mode local
        return self.local_reranker.compute_score(
            sentence_pairs, normalize=normalize, max_length=max_length
        )


# =========================================================
# QDRANT
# =========================================================
def _load_qdrant_client() -> QdrantClient:
    qdrant_cfg = config["qdrant"]

    api_key_env_name = qdrant_cfg.get("api_key")
    api_key_value = getenv(api_key_env_name) if api_key_env_name else None

    return QdrantClient(
        host=qdrant_cfg["host"],
        port=qdrant_cfg["port"],
        timeout=qdrant_cfg.get("timeout", 120),
        api_key=api_key_value,
        https=qdrant_cfg.get("https", False),
        check_compatibility=qdrant_cfg.get("check_compatibility", False),
    )


# =========================================================
# EMBEDDINGS
# =========================================================
def _load_embedding_api():
    """
    Configure l'API d'embedding si activée.
    Variables d'environnement prioritaires : URL_EMBEDDING_API, EMBEDDING_MODEL, EMBEDDING_API_KEY.
    """
    api_cfg = config.get("embedding_api", {})
    enabled = api_cfg.get("enabled", False)

    base_url = (getenv("URL_EMBEDDING_API") or api_cfg.get("base_url", "")).strip()
    api_model = (getenv("EMBEDDING_MODEL") or api_cfg.get("model", "")).strip()

    api_key_env = api_cfg.get("api_key_env", "")
    api_key = getenv("EMBEDDING_API_KEY") or (getenv(api_key_env) if api_key_env else None)
    if api_key:
        api_key = api_key.strip()
    if api_key == "":
        api_key = None

    timeout = int(api_cfg.get("timeout", 30))
    retry_attempts = int(api_cfg.get("retry_attempts", 1))
    retry_delay = float(api_cfg.get("retry_delay", 1.0))

    if enabled and base_url and api_model:
        try:
            api_embedding = ApiEmbeddingModel(
                base_url=base_url,
                model=api_model,
                api_key=api_key,
                timeout=timeout,
                retry_attempts=retry_attempts,
                retry_delay=retry_delay,
            )
            print(f"✓ Configured embedding API: {api_model} ({base_url})")
            return api_embedding
        except Exception as e:
            print(f"⚠ Failed to initialize embedding API: {e}.")
            return None

    return None


def _load_embedding_model():
    # 1. Essayer l'API d'embedding si activée
    api_embedding = _load_embedding_api()
    if api_embedding is not None:
        return api_embedding

    # 2. Sinon, charger le modèle local (si l'embedding local est activé)
    embedding_local_enabled = config.get("model", {}).get("embedding_local", True)
    if not embedding_local_enabled:
        raise ValueError(
            "embedding_api is disabled and embedding_local is false. "
            "No embedding backend is available."
        )

    model_embedding = getenv("EMBEDDING_MODEL") or config["model"]["embedding"]
    model_embedding = model_embedding.strip()

    if "bge-m3" in model_embedding.lower():
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel(model_embedding, use_fp16=True)
        print(f"✓ Loaded hybrid embedding model: {model_embedding}")
        return model

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_embedding)
    print(f"✓ Loaded dense embedding model: {model_embedding}")
    return model


# =========================================================
# RERANKER LOADER
# =========================================================
def _load_reranker_model() -> BaseReranker:
    """
    Charge le reranker.
    Priorité à l'API Albert si configurée et activée.
    Sinon, fallback sur le modèle local.

    Les paramètres sensibles (URL, modèle, clé API) sont lus depuis le fichier
    .env du serveur. Le .env prend priorité sur config.yaml si les deux sont renseignés.
    """
    model_reranker = getenv("RERANKER_MODEL") or config["model"]["reranker"]
    model_reranker = model_reranker.strip()

    # Ne pas charger le modèle local si le reranker local est désactivé
    reranker_local_enabled = config.get("model", {}).get("reranker_local", True)
    if reranker_local_enabled:
        local_reranker = LocalReranker(model_name=model_reranker)
    else:
        local_reranker = None

    api_cfg = config.get("reranker_api", {})
    enabled = api_cfg.get("enabled", False)

    # Paramètres API lus en priorité depuis le .env, avec fallback sur config.yaml
    base_url = (getenv("URL_RERANKER_API") or api_cfg.get("base_url", "")).strip()
    api_model = (getenv("RERANKER_MODEL") or api_cfg.get("model", "")).strip()

    api_key_env = api_cfg.get("api_key_env", "")
    api_key = getenv("RERANKER_API_KEY") or (getenv(api_key_env) if api_key_env else None)
    if api_key:
        api_key = api_key.strip()
    if api_key == "":
        api_key = None

    timeout = int(api_cfg.get("timeout", 30))
    retry_attempts = int(api_cfg.get("retry_attempts", 1))
    retry_delay = float(api_cfg.get("retry_delay", 1.0))
    max_consecutive_errors = int(api_cfg.get("max_consecutive_errors", 3))

    if enabled and base_url and api_model:
        # Le reranker local a déjà été conditionné par reranker_local au début de la fonction

        try:
            api_reranker = ApiReranker(
                base_url=base_url,
                model=api_model,
                api_key=api_key,
                timeout=timeout,
                retry_attempts=retry_attempts,
                retry_delay=retry_delay,
            )
            print(f"✓ Configured Albert API reranker: {api_model} ({base_url})")

            if local_reranker is not None:
                return RerankerRouter(
                    api_reranker=api_reranker,
                    local_reranker=local_reranker,
                    max_consecutive_errors=max_consecutive_errors,
                )
            return api_reranker
        except Exception as e:
            if local_reranker is None:
                raise ValueError(
                    f"reranker_api is enabled but reranker_local is false and API init failed: {e}"
                ) from e
            print(f"⚠ Failed to initialize Albert API reranker: {e}. Using local reranker.")
            return local_reranker

        if local_reranker is None:
            raise ValueError("reranker_api is disabled and reranker_local is false. No reranker backend available.")

    print(f"✓ Loaded local reranker: {model_reranker}")
    return local_reranker


# =========================================================
# CAPABILITIES SAFE (BGE-M3 FIX)
# =========================================================
def _detect_model_capabilities(model) -> dict[str, Any]:
    capabilities = {
        "has_dense": False,
        "has_sparse": False,
        "dense_dim": None,
    }

    test_text = "test"

    try:
        result = model.encode(
            [test_text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        # ---------------- bge-m3 dict output ----------------
        if isinstance(result, dict):
            dense = result.get("dense_vecs")

            if dense is not None and len(dense) > 0:
                vec = dense[0]
                capabilities["has_dense"] = True
                capabilities["dense_dim"] = len(vec)

            if result.get("lexical_weights") is not None:
                capabilities["has_sparse"] = True

            return capabilities

        # ---------------- numpy / list / tensor ----------------
        if result is not None:

            # numpy array safe check
            try:
                import numpy as np
                if isinstance(result, np.ndarray):
                    if result.size > 0:
                        vec = result[0]
                        capabilities["has_dense"] = True
                        capabilities["dense_dim"] = len(vec)
                        return capabilities
            except Exception:
                pass

            # list / tensor fallback
            try:
                if hasattr(result, "__len__") and len(result) > 0:
                    vec = result[0]

                    # si déjà vector 1D
                    if isinstance(vec, (float, int)):
                        vec = result

                    capabilities["has_dense"] = True
                    capabilities["dense_dim"] = len(vec)
            except Exception:
                pass

        return capabilities

    except Exception as e:
        raise ValueError(f"Embedding capability detection failed: {e}") from e


# =========================================================
# INIT
# =========================================================
client = _load_qdrant_client()

EMBEDDING_MODEL_NAME = (getenv("EMBEDDING_MODEL") or config["model"]["embedding"]).strip()
RERANKER_MODEL_NAME = (getenv("RERANKER_MODEL") or config["model"]["reranker"]).strip()

EMBEDDING_MODEL = _load_embedding_model()
RERANKER_MODEL = _load_reranker_model()

MODEL_CAPABILITIES = _detect_model_capabilities(EMBEDDING_MODEL)

model = EMBEDDING_MODEL
reranker = RERANKER_MODEL

COLLECTION = config["collection"]["name"]

BATCH_SIZE = int(config["indexing"].get("batch_size", 5))
BATCHSIZE = BATCH_SIZE

DOCUMENTS_PATH = config["indexing"].get("documents_path", "documents")

SEARCH_LIMIT = int(config.get("search", {}).get("limit", 3))
CANDIDATE_MULTIPLIER = int(config.get("search", {}).get("candidate_multiplier", 10))
MIN_CANDIDATES = int(config.get("search", {}).get("min_candidates", 30))
SCROLL_LIMIT_PER_DOC = int(config.get("search", {}).get("scroll_limit_per_doc", 10000))
WINDOW_RADIUS = int(config.get("search", {}).get("window_radius", 4))

FULL_DOCUMENT_CHUNK_THRESHOLD = int(
    config.get("search", {}).get("full_document_chunk_threshold", 12)
)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


print(
    "✓ Embedding model capabilities:",
    {
        "has_dense": MODEL_CAPABILITIES["has_dense"],
        "has_sparse": MODEL_CAPABILITIES["has_sparse"],
        "dense_dim": MODEL_CAPABILITIES["dense_dim"],
    },
)

print(f"✓ Reranker ready: {RERANKER_MODEL_NAME}")
