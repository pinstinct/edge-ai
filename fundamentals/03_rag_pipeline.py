""" Toy Documents Corpus """
import os
import pprint
import re

import numpy as np
import requests

DOCS = [
    "Foundry Local provides an OpenAI-compatible local inference endpoint.",
    "Retrieval Augmented Generation improves answer grounding by injecting relevant context.",
    "Edge AI reduce latency and preserves privacy via local execution.",
    "Small Language Models can offer competitive quality with lower resources usage.",
    "Vector similarity search retrieves semantically relevant documents."
]

""" Connection, Model Selection & Embedding Init """
# Native Foundry Local SDK preferred; fall back to explicit BASE_URL if provided
os.environ.setdefault("FOUNDRY_LOCAL_ALIAS", "phi-4-mini")
alias = os.getenv("FOUNDRY_LOCAL_ALIAS", os.getenv("TARGET_MODEL", "phi-4-mini"))
base_url_env = os.getenv("BASE_URL", "").strip()
manager = None
client = None
endpoint = None


def _canonicalize(model_id: str) -> str:
    """Remove CUDA suffix and version tags from model name."""
    b = model_id.split(":")[0]
    return re.sub(r"-cuda.*", "", b)


try:
    if base_url_env:
        # Allow user override; normalize by removing trailing / and optional /v1
        root = base_url_env.rstrip("/")
        if root.endswith("/v1"):
            root = root[:3]
        endpoint = root
        print(f"[INFO] Using explicit BASE_URL override: {endpoint}")
    else:
        from foundry_local import FoundryLocalManager

        manager = FoundryLocalManager(alias)
        # Manager endpoint already includes /v1 - remove it for our base
        raw_endpoint = manager.endpoint.rstrip("/")
        if raw_endpoint.endswith("/v1"):
            endpoint = raw_endpoint[:-3]
        else:
            endpoint = raw_endpoint
        print(
            f"[OK] Foundry Local manager endpoint: {manager.endpoint} | base={endpoint} | alias={alias}")

    # Probe models list (endpoint does NOT include /v1 here)
    models_resp = requests.get(endpoint + "/v1/models", timeout=5)
    models_resp.raise_for_status()
    payload = models_resp.json() if models_resp.headers.get("content-type", "").startswith(
        "application/json") else {}
    data = payload.get("data", []) if isinstance(payload, dict) else []
    ids = [m.get("id") for m in data if isinstance(m, dict)]

    # Select best matching model
    chosen = None
    if alias in ids:
        chosen = alias
    else:
        for mid in ids:
            if _canonicalize(mid) == _canonicalize(alias):
                chosen = mid
                break
    if not chosen and ids:
        chosen = ids[0]
    model_name = chosen or alias

    # Initialize OpenAI client
    from openai import OpenAI as _OpenAI

    client = _OpenAI(
        base_url=endpoint + "/v1",  # OpenAI client needs full base URL with /v1
        api_key=(getattr(manager, "api_key", None) or os.getenv("API_KEY") or "not-needed")
    )
    print(f"[OK] Model resolved: {model_name} (total_models={len(ids)})")
except Exception as e:
    print(f"[ERROR] Failed to initialize Foundry Local client:", e)
    client = None
    model_name = alias

# Expose BASE for downstream compatibility (without /v1)
BASE = endpoint

# Embeddings setup
embed_model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
try:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embed_model_name)
    doc_emb = embedder.encode(DOCS, convert_to_numpy=True, normalize_embeddings=True)
    print(f"[OK] Embedded {len(DOCS)} docs using {embed_model_name} shape={doc_emb.shape}")
except Exception as e:
    print(f"[ERROR] Embedding init failed:", e)

try:
    import openai as _openai

    openai_version = getattr(_openai, "__version__", "unknown")
    print("OpenAI SDK version:", openai_version)
except Exception:
    openai_version = "unknown"

if client is None:
    print("\nNEXT: Sstart/verify service then re-run this cell:")
    print("    foundry service start")
    print("    foundry model run phi-4-mini")
    print("    (optional) set BASE_URL=http://127.0.0.1:57127")

""" Retrieve Function (Vector Similarity) """

# def retrieve(query, k=3):
#     q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
#     sims = doc_emb @ q
#     return sims.argsort()[::-1][:k]


""" SDK-Based Generation & Answer Helper """


# SDK-based generation (Foundry Local manager + OpenAI client methods)

def _strip_model_name(name: str) -> str:
    """Strip CUDA suffix and version tags from model name."""
    base = name.split(":")[0]
    base = re.sub(r"-cuda.*", "", base)
    return base


# Use the actual resolved model name from connection cell
RAW_MODEL = model_name
ALT_MODEL = _strip_model_name(RAW_MODEL)


def _try_via_client(messages, prompt, model_id: str, max_tokens=220, temperature=0.2):
    """Try generation response using OpenAI client with multiple fallback routes."""
    attempts = []

    # 1. Try chat.completions endpoint (preferred for chat models)
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        content = resp.choices[0].message.content
        attempts.append(("chat.completions", 200, (content or "")[:160]))
        if content and content.strip():
            return content, attempts
    except Exception as e:
        attempts.append(("chat.completions", None, str(e)[:160]))

    # 2. Try legacy completions endpoint
    try:
        comp = client.completions.create(
            model=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        txt = comp.choices[0].text if comp.choices else ""
        attempts.append(("completions", 200, (txt or "")[:160]))
        if txt and txt.strip():
            return txt, attempts
    except Exception as e:
        attempts.append(("completions", None, str(e)[:160]))

    return None, attempts


def retrieve(query, k=3):
    """Retrieve top-k most similar documents using cosine similarity."""
    if embedder is None or doc_emb is None:
        raise RuntimeError("Embeddings not initialized.")
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    scores = doc_emb @ q_emb
    idxs = np.argsort(scores)[::-1][:k]
    return idxs


def answer(query, k=3, max_tokes=220, temperature=0.2, try_alternate=True):
    """
    Answer a query using RAG pipeline:
    1. Retrieve relevant documents using vector similarity
    2. Generate grounded response using Foundry Local model via OpenAI SDK

    :param query: User question
    :param k: Number of documents to retrieve
    :param max_tokes: Maximum tokens for generation
    :param temperature: Sampling temperature
    :param try_alternate: Whether to try alternate model name on failure
    :return: Dictionary with query, answer, docs, context, route, and tried attempts
    """
    if client is None:
        raise RuntimeError(
            "Model client not initialized. Re-run connection cell after starting Foundry Local")
    if embedder is None or doc_emb is None:
        raise RuntimeError("Embeddings not initialized.")

    # Retrieve relevant documents
    idxs = retrieve(query, k=k)
    context = "\n".join(f"Doc {i}: {DOCS[i]}" for i in idxs)

    # Construct grounded generation prompt
    system_content = "Use Only provided context. If insufficient, Say 'I am not sure'"
    user_content = f"Context:\n{context}\n\nQuestion: {query}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    prompt = f"System: {system_content}\n{user_content}\nAnswer:"

    # Try generation with primary model
    tried = []
    ans, attempts = _try_via_client(messages, prompt, RAW_MODEL, max_tokens=max_tokes,
                                    temperature=temperature)
    tried.append({"model": RAW_MODEL, "attempts": attempts})

    if ans and ans.strip():
        return {
            "query": query,
            "answer": ans.strip(),
            "docs": idxs.tolist(),
            "context": context,
            "route": "chat-first",
            "tried": tried,
        }

    # Try alternate model name if available
    if try_alternate and ALT_MODEL != RAW_MODEL:
        ans2, attempts2 = _try_via_client(messages, prompt, ALT_MODEL, max_tokens=max_tokes,
                                          temperature=temperature)
        tried.append({"model": ALT_MODEL, "attempts": attempts2})
        if ans2 and ans2.strip():
            return {
                "query": query,
                "anser": ans2.strip(),
                "docs": idxs.tolist(),
                "context": context,
                "route": "chat-alt",
                "tried": tried,
            }

    # All routes failed
    return {
        "query": query,
        "answer": "I am not sure. (All SDK routes failed)",
        "docs": idxs.tolist(),
        "context": context,
        "route": "failed",
        "tried": tried,
    }


print("[INFO] SDK generation mode active.")
print(f"    RAW_MODEL = {RAW_MODEL}")
print(f"    ALT_MODEL = {ALT_MODEL}")


# Self-test cell: validates connectivity, embeddings, and answer() basic functionality (SDK mode)

def rag_self_test(sample_query: str = "Why use RAG with local inference?", expect_docs: int = 3):
    report = {"base": BASE, "raw_model": RAW_MODEL, "alt_model": ALT_MODEL}
    if not BASE:
        report["error"] = "BASE not resolved"
        return report
    if embedder is None or doc_emb is None:
        report["error"] = "Embeddings not initialized"
        return report
    if getattr(doc_emb, "shape", (0,))[0] != len(DOCS):
        report[
            "warning_embeddings"] = f"doc_emb count {getattr(doc_emb, 'shape', ('?'))} mismatch DOCS {len(DOCS)}"

    try:
        idxs = retrieve(sample_query, k=expect_docs)
        report["retrieved_indices"] = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
    except Exception as e:
        report["error_retrieve"] = str(e)
        return report

    try:
        ans = answer(sample_query, k=expect_docs, max_tokes=80, temperature=0.2)
        report["route"] = ans.get("route")
        report["answer_preview"] = ans.get("answer", "")[:160]
        if ans.get("route") == "failed":
            report["warning_generation"] = "All SDK routes failed for sample query"
    except Exception as e:
        report["error_generation"] = str(e)
    return report


pprint.pprint(rag_self_test())

""" Batch Query Smoke Test """
queries = [
    "Why use RAG with local inference?",
    "What dose vector similarity search do?",
    "Explain privacy benefits.",
]

last_result = None

for q in queries:
    try:
        r = answer(q)
        last_result = r
        print(f"Q: {q}\nA: {r['answer']}\nDocs: {r['docs']}\n---")
    except Exception as e:
        print(f"Failed answering {q}: {e}")

print(last_result)
print("===")

""" Single Answer Convenience Call """
result = answer("Why use RAG with local inference?")
print(result)
