import math
import os
import time

# PyArrow, TensorFlow의 내부 종속성을 로드할 수 있는 라이브러리
import numpy as np
from datasets import Dataset
from foundry_local import FoundryLocalManager
from foundry_local.models import FoundryModelInfo
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas import RunConfig, evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision

# 충돌 유발 가능성이 높은 SentenceTransformer를 맨 마지막에 임포트
from sentence_transformers import SentenceTransformer  # for Embedding

""" Toy Corpus & QA Ground Truth """
DOCS = [
    "Foundry Local exposes a local OpenAI-compatible endpoint.",
    "Rag retrieves relevant context snippets before generation.",
    "Local inference improves privacy and reduces latency.",
]

QUESTIONS = [
    "What advantage does local inference offer?",
    "How does RAG improve grounding?",
]

GROUND_TRUTH = [
    "It reduces latency and preserves privacy.",
    "It adds retrieved context snippets for factual grounding."
]

""" Service init, Embeddings & Safety Patch """
# --- Safe monkeypatch for potential null promptTemplate field (schema drift guard) ---
_original_from_list_response = FoundryModelInfo.from_list_response


def _safe_from_list_response(response):
    try:
        if isinstance(response, dict) and response.get("promptTemplate") is None:
            response["promptTemplate"] = {}
    except Exception as e:  # pragma: no cover
        print(f"Warning normalizing promptTemplate: {e}")
    return _original_from_list_response(response)


if getattr(FoundryModelInfo.from_list_response, "__name__", "") != "_safe_from_list_response":
    FoundryModelInfo.from_list_response = staticmethod(_safe_from_list_response)
# --- End monkeypatch

alias = os.getenv("FOUNDRY_LOCAL_ALIAS", "phi-3.5-mini")
manager = FoundryLocalManager(alias)
print(f"Service running: {manager.is_service_running()} | Endpoint: {manager.endpoint}")
print(f"Cached models: {manager.list_cached_models()}")
model_info = manager.get_model_info(alias)
model_id = model_info.id
print(f"Using model id: {model_id}")

# OpenAI-compatible client
client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key or "not-needed")

# 텍스트를 숫자 형태(벡터)로 변환
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
doc_emb = embedder.encode(DOCS, convert_to_numpy=True, normalize_embeddings=True)

""" Retriever Function (검색 함수) """
def retrieve(query, k=2):  # 상위 k개 문서 반환
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]  # query의 임베딩 벡터
    sims = doc_emb @ q  # @연산자: 행렬 곱, sims: 유사도 점수 리스트
    return [DOCS[i] for i in sims.argsort()[::-1][:k]]  # argsort(): sims를 점수가 낮은 것부터 정렬했을 때 인덱스 반환, [::1]: 리스트 역순으로 변환, [:k]: 상위 k개 slice


""" Generate Function (생성 함수) """
# 제한된 프롬프트로 로컬 모델 호출
def generate(query, contexts):
    ctx = "\n".join(contexts)
    messages = [
        {"role": "system", "content": "Answer using ONLY the provided context."},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}
    ]
    resp = client.chat.completions.create(model=model_id, messages=messages, max_tokens=120,
                                          temperature=0.1)  # 낮은 온도(temperature)는 창의성보다 충실한 추출을 선호
    return resp.choices[0].message.content.strip()


""" Fallback Client Initialization (폴백 클라이언트 초기화) """
try:
    client
except NameError:
    client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key or "not-needed")
    print(f"Initialized OpenAI-compatible client (late init).")

""" Evaluation Loop & Metics (평가 루프 및 메트릭) """
# Build evaluation dataset with required columns (including 'reference' for context_precision)
records = []
for q, gt in zip(QUESTIONS, GROUND_TRUTH):
    ctxs = retrieve(q)
    ans = generate(q, ctxs)
    records.append({
        "question": q,
        "answer": ans,
        "contexts": ctxs,
        "ground_truths": [gt],
        "reference": gt,
    })

# Ragas가 평가에 사용할 LLM 설정
ragas_llm = ChatOpenAI(model=model_id, base_url=manager.endpoint,
                       api_key=manager.api_key or "not-needed", temperature=0.0, timeout=60)


# Ragas가 평가에 사용할 임베딩 모델 클래스 정의
class LocalEmbedding:
    # 임베딩 후 리스트로 변환
    def embed_documents(self, texts):
        return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    # 임베딩 후 단일 벡터 리스트로 변환
    def embed_query(self, text):
        return embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[
            0].tolist()


# Fast mode: only answer_relevance unless RAG_FAST=0
FAST_MODE = os.getenv("RAG_FAST", "1") == "1"  # 가장 빠른 answer_relevancy만 계산할지 결정
# answer_relevancy: 답변 관련성
# faithfulness: 충실도, 답변이 사실만 답고 있는지
# context_precision: 컨텍스트 정확도, 검색 효율성
metrics = [answer_relevancy] if FAST_MODE else [answer_relevancy, faithfulness, context_precision]

base_timeout = 45 if FAST_MODE else 120

ds = Dataset.from_list(records)  # 리스트 형태의 records를 데이터셋 객체로 변환
print("Evaluation dataset columns:", ds.column_names)
print("Metrics to compute:", [m.name for m in metrics])

results_dict = {}
for metric in metrics:
    t0 = time.time()
    try:
        cfg = RunConfig(timeout=base_timeout, max_workers=1)
        partial = evaluate(ds, metrics=[metric], llm=ragas_llm, embeddings=LocalEmbedding(),
                           run_config=cfg, show_progress=False)  # evaluate(): Ragas 핵심 함수
        raw_val = partial[metric.name]
        if isinstance(raw_val, list):
            numeric = [v for v in raw_val if isinstance(v, (int, float))]
            score = float(np.nanmean(numeric)) if numeric else math.nan
        else:
            score = float(raw_val)
        results_dict[metric.name] = score
    except Exception as e:
        results_dict[metric.name] = math.nan
        print(f"Metric {metric.name} failed: {e}")
    finally:
        print(f"{metric.name} finished in {time.time() - t0:.1f}s -> {results_dict[metric.name]}")

print("RAG evaluation results:", results_dict)
