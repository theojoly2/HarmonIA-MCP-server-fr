from pathlib import Path
from hashlib import sha256
import os
import re
import codecs
import json
import csv
from io import StringIO
from typing import Any
import pymupdf4llm

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseVector,
)

try:
    from .load_documents import config as cf
except ImportError:
    import config as cf

client = cf.client
model = cf.model
COLLECTION = cf.COLLECTION
BATCH_SIZE = cf.BATCH_SIZE

# ─── Configuration des résumés ─────────────────────────────────────────────
SUMMARY_CFG = getattr(cf, "config", {}).get("summary", {})
SUMMARY_ENABLED = bool(SUMMARY_CFG.get("enabled", False))
SUMMARY_LANG = str(SUMMARY_CFG.get("lang", "fr"))
SUMMARY_LLM_CFG = SUMMARY_CFG.get("llm", {})
SUMMARY_MAX_INPUT_CHARS = int(SUMMARY_LLM_CFG.get("max_input_chars", 800_000))
SUMMARY_MIN_OUTPUT_CHARS = int(SUMMARY_LLM_CFG.get("min_output_chars", 200))
SUMMARY_MAX_OUTPUT_CHARS = int(SUMMARY_LLM_CFG.get("max_output_chars", 400))
SUMMARY_FALLBACK_MAX_CHARS = int(SUMMARY_CFG.get("fallback", {}).get("max_chars", 400))
SUMMARY_SYSTEM_PROMPT = SUMMARY_LLM_CFG.get("system_prompt", "")

def _load_summary_llm_config():
    """Charge la configuration LLM pour les résumés (.env prioritaire sur config.yaml)."""
    summary_cfg = getattr(cf, "config", {}).get("summary", {})
    base_url = (os.getenv("URL_LLM_API") or summary_cfg.get("base_url", "")).strip()
    model = (os.getenv("LLM_MODEL") or summary_cfg.get("model", "")).strip()

    api_key_env = summary_cfg.get("api_key_env", "")
    api_key = os.getenv("LLM_API_KEY") or (os.getenv(api_key_env) if api_key_env else None)
    if api_key:
        api_key = api_key.strip()
    if api_key == "":
        api_key = None

    return base_url, model, api_key


_LLM_CLIENT = None
_LLM_MODEL = None
if SUMMARY_ENABLED:
    try:
        from openai import OpenAI

        _URL_API, _LLM_MODEL, _LLM_API_KEY = _load_summary_llm_config()
        if _URL_API and _LLM_API_KEY and _LLM_MODEL:
            _LLM_CLIENT = OpenAI(base_url=_URL_API, api_key=_LLM_API_KEY)
            print(f"✓ Summary LLM configured: {_LLM_MODEL} ({_URL_API})")
        else:
            print(
                "[!] Summary LLM enabled but missing LLM_API_KEY / URL_LLM_API / LLM_MODEL. "
                "Falling back to native descriptions."
            )
    except Exception as e:
        print(f"[!] Failed to initialize summary LLM client: {e}")

XML_DECL_RE = re.compile(br'<\?xml[^>]+encoding\s*=\s*["\']([^"\']+)', re.IGNORECASE)
TEXT_BOMS = (
    codecs.BOM_UTF8,
    codecs.BOM_UTF16_LE,
    codecs.BOM_UTF16_BE,
    codecs.BOM_UTF32_LE,
    codecs.BOM_UTF32_BE,
)
TEXT_WHITELIST = set(b"\t\n\r")
PRINTABLE_ASCII = set(range(32, 127))

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# ─── Helpers extraction native de description ──────────────────────────────

def _truncate_text(text: str, max_chars: int) -> str:
    """Tronque proprement un texte à max_chars caractères."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Coupure au dernier espace avant la limite pour éviter de couper un mot
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip() + "…"


def _clean_fallback_text(text: str) -> str:
    """Nettoie un texte brut pour en faire un fallback descriptif."""
    # Supprime les lignes vides répétées et les espaces superflus
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = " ".join(lines)
    # Supprime les artefacts fréquents (URL, tags XML bruts, etc.)
    joined = re.sub(r"\s+", " ", joined)
    return joined.strip()


def _extract_description_from_json(text: str) -> str:
    """Extrait la description d'un document JSON/JSON-LD."""
    try:
        data = json.loads(text)
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    for key in ("description", "rdfs:comment", "comment", "abstract", "dcterms:description", "dc:description"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict):
                # JSON-LD avec @value
                val = first.get("@value", "")
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return ""


def _extract_description_from_ttl(text: str) -> str:
    """Extrait la description d'un document Turtle/OWL."""
    try:
        import rdflib
    except ImportError:
        return ""
    try:
        g = rdflib.Graph()
        g.parse(data=text, format="turtle")
    except Exception:
        return ""

    predicates = [
        rdflib.URIRef("http://purl.org/dc/terms/description"),
        rdflib.URIRef("http://purl.org/dc/elements/1.1/description"),
        rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#comment"),
        rdflib.URIRef("http://www.w3.org/2004/02/skos/core#definition"),
        rdflib.URIRef("http://purl.org/dc/terms/abstract"),
    ]

    # Cherche d'abord sur owl:Ontology
    for s, p, o in g.triples((None, rdflib.RDF.type, rdflib.OWL.Ontology)):
        for pred in predicates:
            for obj in g.objects(s, pred):
                val = str(obj)
                if val.strip():
                    return val.strip()

    # Sinon, première description trouvée dans le graphe
    for pred in predicates:
        for obj in g.objects(None, pred):
            val = str(obj)
            if val.strip():
                return val.strip()

    return ""


def _extract_description_from_xml(text: str) -> str:
    """Extrait la description d'un document XML/XMI."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
    except Exception:
        return ""

    # Namespaces courants
    namespaces = {
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "dcterms": "http://purl.org/dc/terms/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "uml": "http://schema.omg.org/spec/UML/2.1",
        "xmi": "http://schema.omg.org/spec/XMI/2.1",
    }

    tags = [
        ".//rdfs:comment",
        ".//skos:definition",
        ".//dcterms:description",
        ".//dc:description",
        ".//{http://www.sparxsystems.com/profiles/thecustomprofile/1.0}definition",
        ".//{http://www.sparxsystems.com/profiles/TerminologyProfile/1.0}definition",
    ]

    for tag in tags:
        try:
            if tag.startswith(".//{http"):
                elems = root.findall(tag)
            else:
                elems = root.findall(tag, namespaces)
            for elem in elems:
                if elem.text and elem.text.strip():
                    return elem.text.strip()
        except Exception:
            continue

    return ""


def _extract_description_from_sql(text: str) -> str:
    """Extrait la description des commentaires de header SQL."""
    lines = []
    for raw_line in text.splitlines()[:30]:
        line = raw_line.strip()
        if line.startswith("--"):
            comment = line[2:].strip("- ").strip()
            if comment:
                lines.append(comment)
        elif line.startswith("/*") and not line.endswith("*/"):
            continue
        elif lines and line == "*/":
            break
        elif not line or line.startswith("SET") or line.startswith("CREATE"):
            if lines:
                break
    description = " ".join(lines)
    return description.strip()


def extract_native_description(filepath: Path, text: str) -> str:
    """
    Extrait une description native du fichier selon son type.
    Si aucune description structurée n'est trouvée, retourne un texte descriptif court.
    """
    ext = filepath.suffix.lower()
    description = ""

    if ext in (".json", ".jsonld"):
        description = _extract_description_from_json(text)
    elif ext in (".ttl", ".owl"):
        description = _extract_description_from_ttl(text)
    elif ext in (".xml", ".xmi"):
        description = _extract_description_from_xml(text)
    elif ext in (".sql", ".ddl"):
        description = _extract_description_from_sql(text)

    if description and len(description.strip()) >= 20:
        return _truncate_text(description.strip(), SUMMARY_FALLBACK_MAX_CHARS)

    # Fallback : 400 premiers caractères descriptifs du texte
    cleaned = _clean_fallback_text(text)
    return _truncate_text(cleaned, SUMMARY_FALLBACK_MAX_CHARS)


def generate_llm_summary(filename: str, full_text: str) -> str:
    """
    Appelle le LLM configuré pour générer un résumé du document.
    Non bloquant : retourne une chaîne vide en cas d'échec.
    """
    if _LLM_CLIENT is None or not _LLM_MODEL:
        return ""

    if not full_text.strip():
        return ""

    truncated = False
    text_input = full_text
    if len(full_text) > SUMMARY_MAX_INPUT_CHARS:
        text_input = full_text[:SUMMARY_MAX_INPUT_CHARS]
        truncated = True
        print(
            f"[~] '{filename}' truncated for LLM summary: "
            f"{len(full_text)} → {SUMMARY_MAX_INPUT_CHARS} chars"
        )

    system_prompt = SUMMARY_SYSTEM_PROMPT.format(lang=SUMMARY_LANG)
    user_message = (
        f"Voici le contenu du document '{filename}'"
        + (" (tronqué, suite non disponible)" if truncated else "")
        + f" :\n\n{text_input}\n\nRésume ce document selon les instructions."
    )

    try:
        response = _LLM_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        summary = response.choices[0].message.content or ""
        summary = summary.strip()
        if len(summary) < SUMMARY_MIN_OUTPUT_CHARS:
            # Trop court : probablement invalide, on retombe en extraction native
            print(f"[!] LLM summary too short for '{filename}' ({len(summary)} chars). Using native fallback.")
            return ""
        if len(summary) > SUMMARY_MAX_OUTPUT_CHARS:
            summary = _truncate_text(summary, SUMMARY_MAX_OUTPUT_CHARS)
        return summary
    except Exception as e:
        print(f"[!] LLM summary generation failed for '{filename}': {e}")
        return ""


def get_doc_summary(filename: str, filepath: Path, full_text: str) -> str:
    """
    Renvoie le résumé d'un document.
    Si summary.enabled=true et LLM disponible → génération LLM.
    Sinon → extraction native (description du fichier ou texte descriptif).
    """
    if SUMMARY_ENABLED and _LLM_CLIENT is not None:
        llm_summary = generate_llm_summary(filename, full_text)
        if llm_summary:
            return llm_summary
        print(f"[~] LLM summary failed for '{filename}', falling back to native description.")

    return extract_native_description(filepath, full_text)

CHUNK_SIZE = int(
    getattr(cf, "config", {}).get("chunking", {}).get("chunk_size", 4000)
    if hasattr(cf, "config")
    else 4000
)

CHUNK_OVERLAP = int(
    getattr(cf, "config", {}).get("chunking", {}).get("chunk_overlap", 400)
    if hasattr(cf, "config")
    else 400
)


def detect_model_capabilities(model) -> dict[str, Any]:
    """
    Detect if the loaded model supports dense vectors, sparse vectors, or both.
    """
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

        if isinstance(result, dict):
            dense_vecs = result.get("dense_vecs")
            lexical_weights = result.get("lexical_weights")

            if dense_vecs is not None:
                first_dense = dense_vecs[0] if len(dense_vecs) > 0 else None
                if first_dense is not None:
                    capabilities["has_dense"] = True
                    capabilities["dense_dim"] = len(first_dense)

            if lexical_weights is not None:
                capabilities["has_sparse"] = True

            print(
                f"Model capabilities detected: "
                f"dense={capabilities['has_dense']}, "
                f"sparse={capabilities['has_sparse']}, "
                f"dense_dim={capabilities['dense_dim']}"
            )
            return capabilities

    except Exception:
        pass

    try:
        dense = model.encode([test_text])
        first_dense = dense[0] if hasattr(dense, "__len__") else dense
        capabilities["has_dense"] = True
        capabilities["dense_dim"] = len(first_dense)
        print(
            f"Model capabilities detected: dense=True, sparse=False, "
            f"dense_dim={capabilities['dense_dim']}"
        )
        return capabilities
    except Exception as e:
        raise ValueError(f"Could not detect model capabilities: {e}")


def detect_bom_encoding(raw: bytes) -> str | None:
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith(codecs.BOM_UTF32_LE):
        return "utf-32-le"
    if raw.startswith(codecs.BOM_UTF32_BE):
        return "utf-32-be"
    if raw.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    if raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    return None


def detect_xml_decl_encoding(raw: bytes) -> str | None:
    head = raw[:2048]
    match = XML_DECL_RE.search(head)
    if match:
        try:
            return match.group(1).decode("ascii").lower()
        except Exception:
            return None
    return None


def is_probably_binary(filepath: Path, chunk_size: int = 4096) -> bool:
    raw = filepath.read_bytes()
    if not raw:
        return False

    sample = raw[:chunk_size]

    if any(sample.startswith(bom) for bom in TEXT_BOMS):
        return False

    if b"\x00" in sample:
        return True

    odd = 0
    for b in sample:
        if b in TEXT_WHITELIST or b in PRINTABLE_ASCII:
            continue
        if b >= 128:
            continue
        odd += 1

    return (odd / len(sample)) > 0.30


def guess_text_encodings(raw: bytes) -> list[str]:
    candidates = []

    bom_encoding = detect_bom_encoding(raw)
    if bom_encoding:
        candidates.append(bom_encoding)

    xml_decl_encoding = detect_xml_decl_encoding(raw)
    if xml_decl_encoding:
        candidates.append(xml_decl_encoding)

    candidates.extend(
        [
            "utf-8",
            "utf-8-sig",
            "utf-16",
            "utf-16-le",
            "utf-16-be",
            "utf-32",
            "utf-32-le",
            "utf-32-be",
            "cp1252",
            "latin-1",
        ]
    )

    deduped = []
    seen = set()
    for enc in candidates:
        if enc and enc not in seen:
            deduped.append(enc)
            seen.add(enc)

    return deduped


def read_text_document(filepath: Path) -> tuple[str, str]:
    raw = filepath.read_bytes()

    if is_probably_binary(filepath):
        raise ValueError(f"{filepath.name} appears to be a binary/non-text file")

    encodings_to_try = guess_text_encodings(raw)

    for encoding in encodings_to_try:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
        except LookupError:
            continue

    raise ValueError(
        f"Unable to decode file {filepath.name}. "
        f"Tried encodings: {', '.join(encodings_to_try)}"
    )


def read_pdf_document(filepath: Path) -> tuple[str, str]:
    """
    Lit un document PDF et le convertit en Markdown en utilisant pymupdf4llm.
    """

    print(f"[~] Converting PDF to Markdown via pymupdf4llm: {filepath.name}")
    md_text = pymupdf4llm.to_markdown(str(filepath))
    return md_text, "pdf-to-markdown"


def optimize_json_preserving_standards(data: Any) -> str:
    def transform(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, str):
                    out[k] = v.strip()
                else:
                    out[k] = transform(v)
            return out

        if isinstance(obj, list):
            return [transform(item) for item in obj]

        if isinstance(obj, str):
            return obj.strip()

        return obj

    optimized = transform(data)
    return json.dumps(optimized, separators=(",", ":"), ensure_ascii=False)


def optimize_xml_preserving_standards(xml_content: str) -> str:
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_content)

        def strip_whitespace(elem):
            if elem.text is not None:
                elem.text = elem.text.strip() or None
            if elem.tail is not None:
                elem.tail = elem.tail.strip() or None
            for child in list(elem):
                strip_whitespace(child)

        strip_whitespace(root)
        return ET.tostring(root, encoding="unicode", method="xml")
    except Exception:
        compact = re.sub(r">\s+<", "><", xml_content)
        return compact.strip()


def optimize_rdflike_text(content: str) -> str:
    lines = []
    previous_blank = False

    for line in content.splitlines():
        stripped = line.strip()

        if not stripped:
            if not previous_blank:
                lines.append("")
                previous_blank = True
            continue

        previous_blank = False

        if stripped.startswith("@"):
            lines.append(stripped)
            continue

        compressed = re.sub(r"\s+", " ", stripped)
        lines.append(compressed)

    return "\n".join(lines)


def optimize_yaml_content(text: str) -> str:
    try:
        import yaml

        data = yaml.safe_load(text)
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        lines = []
        for line in text.splitlines():
            if not line.strip():
                continue
            lines.append(line.rstrip())
        return "\n".join(lines)


def optimize_csv_content(text: str) -> str:
    try:
        reader = csv.reader(StringIO(text))
        output = StringIO()
        writer = csv.writer(output, lineterminator="\n")

        for row in reader:
            if not row:
                continue
            cleaned = [cell.strip() for cell in row]
            if not any(cleaned):
                continue
            writer.writerow(cleaned)

        return output.getvalue().strip()
    except Exception:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lines.append(",".join(part.strip() for part in stripped.split(",")))
        return "\n".join(lines)


def optimize_html_content(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r">\s+<", "><", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def optimize_markdown_content(text: str) -> str:
    lines = []
    previous_blank = False

    for line in text.splitlines():
        stripped = line.rstrip()

        if not stripped.strip():
            if not previous_blank:
                lines.append("")
                previous_blank = True
            continue

        lines.append(stripped)
        previous_blank = False

    return "\n".join(lines).strip()


def optimize_generic_text(text: str) -> str:
    lines = []
    previous_blank = False

    for line in text.splitlines():
        stripped = line.rstrip()

        if not stripped.strip():
            if not previous_blank:
                lines.append("")
                previous_blank = True
            continue

        lines.append(re.sub(r"\s+", " ", stripped))
        previous_blank = False

    return "\n".join(lines).strip()


def optimize_document_content(filepath: Path, text: str) -> tuple[str, bool]:
    ext = filepath.suffix.lower()

    try:
        if ext == ".json":
            data = json.loads(text)
            optimized = optimize_json_preserving_standards(data)
            return optimized, optimized != text

        if ext in (".xml", ".xmi", ".xsd"):
            optimized = optimize_xml_preserving_standards(text)
            return optimized, optimized != text

        if ext in (".ttl", ".rdf", ".owl", ".n3", ".nt", ".trig"):
            optimized = optimize_rdflike_text(text)
            return optimized, optimized != text

        if ext in (".yaml", ".yml"):
            optimized = optimize_yaml_content(text)
            return optimized, optimized != text

        if ext == ".csv":
            optimized = optimize_csv_content(text)
            return optimized, optimized != text

        if ext in (".html", ".htm", ".xhtml"):
            optimized = optimize_html_content(text)
            return optimized, optimized != text

        # Les PDF convertis en Markdown passent par ici également
        if ext in (".md", ".markdown", ".rst", ".adoc", ".txt", ".pdf"):
            optimized = optimize_markdown_content(text)
            return optimized, optimized != text

        optimized = optimize_generic_text(text)
        if len(optimized) < len(text) * 0.95:
            return optimized, True

        return text, False

    except Exception as e:
        print(f"[!] Optimization failed for {filepath.name}: {e}")
        return text, False


def generate_document_id(filepath: Path) -> str:
    base = str(filepath.resolve())
    return sha256(base.encode("utf-8")).hexdigest()[:24]


def generate_chunk_stable_id(document_id: str, chunk_index: int) -> int:
    base = f"{document_id}:{chunk_index}"
    hash_bytes = sha256(base.encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def split_text_uniformly(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 10)

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            window_start = max(start, end - 300)
            newline_pos = text.rfind("\n", window_start, end)
            space_pos = text.rfind(" ", window_start, end)
            best_cut = max(newline_pos, space_pos)
            if best_cut > start + (chunk_size // 2):
                end = best_cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(start + step, end - overlap)

    return chunks


def setup_collection(capabilities: dict[str, Any]) -> bool:
    collections = client.get_collections().collections
    collection_exists = any(c.name == COLLECTION for c in collections)

    if collection_exists:
        print(f"[!] Collection '{COLLECTION}' already exists.")
        choice = input("Delete and recreate? (y/n): ").strip().lower()
        if choice == "y":
            client.delete_collection(collection_name=COLLECTION)
            print(f"[✓] Deleted collection '{COLLECTION}'")
        else:
            print(f"[~] Using existing collection '{COLLECTION}'")
            return False

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=capabilities["dense_dim"],
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )
        print(
            f"[✓] Created hybrid collection '{COLLECTION}' "
            f"(dense:{capabilities['dense_dim']}, sparse:yes)"
        )

    elif capabilities["has_dense"]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=capabilities["dense_dim"],
                distance=Distance.COSINE,
            ),
        )
        print(
            f"[✓] Created dense-only collection '{COLLECTION}' "
            f"(dense:{capabilities['dense_dim']})"
        )

    elif capabilities["has_sparse"]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={},
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )
        print(f"[✓] Created sparse-only collection '{COLLECTION}'")

    else:
        raise ValueError("Model has neither dense nor sparse capability")

    return True


def get_existing_ids():
    try:
        result = client.scroll(
            collection_name=COLLECTION,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {point.id for point in result[0]}
        print(f"[~] Found {len(existing_ids)} existing documents")
        return existing_ids
    except Exception as e:
        print(f"[!] Failed to retrieve existing IDs: {e}")
        return set()


def _lexical_to_sparse_vector(lexical_weights: dict[Any, Any]) -> SparseVector:
    indices = [int(k) for k in lexical_weights.keys()]
    values = [float(v) for v in lexical_weights.values()]
    return SparseVector(indices=indices, values=values)


def encode_batch(texts: list[str], capabilities: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        result = model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = result.get("dense_vecs", [])
        lexical_weights = result.get("lexical_weights", [])

        for i in range(len(texts)):
            item = {}
            if i < len(dense_vecs):
                dense = dense_vecs[i]
                item["dense"] = dense.tolist() if hasattr(dense, "tolist") else list(dense)
            if i < len(lexical_weights):
                item["sparse"] = _lexical_to_sparse_vector(lexical_weights[i])
            outputs.append(item)

        return outputs

    if capabilities["has_dense"]:
        dense_vecs = model.encode(texts)
        for dense in dense_vecs:
            outputs.append(
                {"dense": dense.tolist() if hasattr(dense, "tolist") else list(dense)}
            )
        return outputs

    if capabilities["has_sparse"]:
        result = model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        lexical_weights = result.get("lexical_weights", [])
        for lw in lexical_weights:
            outputs.append({"sparse": _lexical_to_sparse_vector(lw)})
        return outputs

    raise ValueError("No supported vector output available from model")


def build_point(
    docid: int,
    text: str,
    filename: str,
    encodingused: str,
    vectors: dict[str, Any],
    capabilities: dict[str, Any],
    payloadextra: dict[str, Any] | None = None,
) -> PointStruct:
    payload = {
        "text": text,
        "filename": filename,
        "encoding": encodingused,
    }
    if payloadextra:
        payload.update(payloadextra)

    if capabilities["has_dense"] and capabilities["has_sparse"]:
        return PointStruct(
            id=docid,
            vector={
                DENSE_VECTOR_NAME: vectors["dense"],
                SPARSE_VECTOR_NAME: vectors["sparse"],
            },
            payload=payload,
        )

    if capabilities["has_dense"]:
        return PointStruct(
            id=docid,
            vector=vectors["dense"],
            payload=payload,
        )

    if capabilities["has_sparse"]:
        return PointStruct(
            id=docid,
            vector={SPARSE_VECTOR_NAME: vectors["sparse"]},
            payload=payload,
        )

    raise ValueError("Unable to build point: no vectors available")


def flush_batch(
    batch_docs: list[dict[str, Any]],
    points: list[PointStruct],
    capabilities: dict[str, Any],
) -> int:
    texts = [item["text"] for item in batch_docs]
    encoded_vectors = encode_batch(texts, capabilities)

    for item, vectors in zip(batch_docs, encoded_vectors):
        point = build_point(
            docid=item["docid"],
            text=item["text"],
            filename=item["filename"],
            encodingused=item["encoding"],
            vectors=vectors,
            capabilities=capabilities,
            payloadextra=item.get("payloadextra"),
        )
        points.append(point)
        print(
            f"[~] Loaded {item['filename']} "
            f"(chunk {item['payloadextra'].get('chunk_index', 0) + 1}/"
            f"{item['payloadextra'].get('chunk_count', 1)}) "
            f"({item['encoding']})"
        )

    client.upsert(collection_name=COLLECTION, points=points)
    uploaded = len(points)
    print(f"[✓] Uploaded batch of {uploaded}")

    points.clear()
    batch_docs.clear()

    return uploaded


def index_documents():
    print("Detecting model capabilities...")
    capabilities = cf.MODEL_CAPABILITIES

    is_fresh = setup_collection(capabilities)
    existing_ids = set() if is_fresh else get_existing_ids()

    docs_path = Path(__file__).parent / "documents"

    total_indexed = 0
    total_skipped = 0
    total_nontext = 0
    total_optimized = 0
    total_chunks = 0

    batch_docs = []
    points = []

    if not docs_path.exists():
        print(f"[!] Directory not found: {docs_path}")
        return

    print(f"[~] Scanning documents in {docs_path}")

    def pushdoc(
        filename: str,
        content: str,
        encodingused: str,
        payloadextra: dict[str, Any] | None = None,
    ):
        nonlocal total_skipped

        payloadextra = payloadextra or {}
        document_id = str(payloadextra.get("document_id"))
        chunk_index = int(payloadextra.get("chunk_index", 0))
        docid = generate_chunk_stable_id(document_id, chunk_index)

        if not is_fresh and docid in existing_ids:
            print(f"[~] Skipped duplicate: {filename} chunk={chunk_index}")
            total_skipped += 1
            return

        batch_docs.append(
            {
                "filename": filename,
                "text": content,
                "encoding": encodingused,
                "docid": docid,
                "payloadextra": payloadextra,
            }
        )

    for filepath in docs_path.iterdir():
        if not filepath.is_file():
            continue

        ext = filepath.suffix.lower()

        try:
            if ext == ".pdf":
                text, encodingused = read_pdf_document(filepath)
            else:
                text, encodingused = read_text_document(filepath)

            optimized_text, was_optimized = optimize_document_content(filepath, text)

            if was_optimized:
                total_optimized += 1
                origin_len = len(text)
                optimized_len = len(optimized_text)
                reduction = ((1 - (optimized_len / origin_len)) * 100) if origin_len else 0
                print(
                    f"[~] Optimized {filepath.name}: "
                    f"{origin_len} → {optimized_len} chars "
                    f"({reduction:.1f}% reduction)"
                )

            document_id = generate_document_id(filepath)

            # Génération/extraction du résumé (stocké uniquement sur le chunk 0)
            doc_summary = get_doc_summary(filepath.name, filepath, optimized_text)
            if doc_summary:
                print(f"[~] Summary for {filepath.name}: {doc_summary[:120]}{'…' if len(doc_summary) > 120 else ''}")

            chunks = split_text_uniformly(optimized_text)

            if not chunks:
                continue

            total_chunks += len(chunks)
            print(f"[~] Chunked {filepath.name} into {len(chunks)} chunks")

            for chunk_index, chunk_text in enumerate(chunks):
                payloadextra = {
                    "doctype": "uniform_chunk",
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "is_child_chunk": True,
                    "source_path": str(filepath.resolve()),
                    "source_extension": filepath.suffix.lower(),
                    "document_name": filepath.stem,
                }
                if chunk_index == 0 and doc_summary:
                    payloadextra["doc_summary"] = doc_summary

                pushdoc(
                    filename=filepath.name,
                    content=chunk_text,
                    encodingused=encodingused,
                    payloadextra=payloadextra,
                )

                if len(batch_docs) >= BATCH_SIZE:
                    try:
                        total_indexed += flush_batch(batch_docs, points, capabilities)
                    except Exception as e:
                        print(f"[!] Batch upload failed: {e}")
                        batch_docs.clear()
                        points.clear()

        except ValueError as e:
            print(f"[!] Skipped non-text file {filepath.name}: {e}")
            total_nontext += 1
        except Exception as e:
            print(f"[!] Failed to read {filepath.name}: {e}")

    if batch_docs:
        try:
            total_indexed += flush_batch(batch_docs, points, capabilities)
        except Exception as e:
            print(f"[!] Final batch upload failed: {e}")

    print("=" * 50)
    print("Indexing complete!")
    print(f"[✓] Chunks indexed: {total_indexed}")
    print(f"[✓] Documents optimized: {total_optimized}")
    print(f"[✓] Total chunks created: {total_chunks}")
    if total_skipped > 0:
        print(f"[~] Chunks skipped (duplicates): {total_skipped}")
    if total_nontext > 0:
        print(f"[!] Files skipped (non-text): {total_nontext}")
    print("=" * 50)


if __name__ == "__main__":
    index_documents()
