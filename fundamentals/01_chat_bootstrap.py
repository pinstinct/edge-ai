import os

from foundry_local import FoundryLocalManager
from foundry_local.models import FoundryModelInfo
from openai import OpenAI
from rich import print


"""
1. Foundry Local 설치
brew tap microsoft/foundrylocal
brew install foundrylocal

2. Foundry 명령어
foundry service start
foundry service stop
foundry service status
foundry model ls
"""

""" Manager Initialization & Metadata Patch """
# Monkeypatch(동적 프로그래밍 언어의 런타임 코드를 동적으로 수정하는 행위)
# to tolerate service responses where promptTemplate is null
_original_from_list_response = FoundryModelInfo.from_list_response


def _safe_from_list_response(response):
    try:
        if isinstance(response, dict) and response.get("promptTemplate") is None:
            response["promptTemplate"] = {}
    except Exception as e:
        print(
            f"[yellow]Warning: safe wrapper encountered issue normalizing promptTemplate: {e}[/yellow]")
    return _original_from_list_response(response)


# Apply patch only once
if getattr(FoundryModelInfo.from_list_response, "__name__", "") != "_safe_from_list_response":
    FoundryModelInfo.from_list_response = staticmethod(_safe_from_list_response)

ALIAS = os.getenv("FOUNDRY_LOCAL_ALIAS", "phi-3.5-mini")

# 로컬 모델 런타임과 인터페이스
manager = FoundryLocalManager(ALIAS)
print(f"[bold green]Service running:[/bold green] {manager.is_service_running()}")
print(f"Endpoint: {manager.endpoint}")
print("Cached models:", manager.list_cached_models())
model_id = manager.get_model_info(ALIAS).id
print(f"Using model id: {model_id}")

""" Basic Chat Completion """
# OpenAI 클라이언트를 통해 익숙한 chat completion API 재사용 가능
client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key or 'not-needed')

prompt = "List two benefits of local inference for privacy."
resp = client.chat.completions.create(model=model_id,
                                      messages=[{"role": "user", "content": prompt}])
print(resp.choices[0].message.content)
print()

""" Streaming Chat Completion """
stream = client.chat.completions.create(model=model_id, messages=[
    {"role": "user", "content": "Give a one-sentence definition of edge AI."}], stream=True,
                                        max_tokens=60, temperature=0.4)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="", flush=True)
print()
