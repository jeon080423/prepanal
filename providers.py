"""AI provider abstraction for Synthetic Respondent Studio.

두 가지 호출 경로를 지원한다:
1. 로컬 개발 환경: ~/workspace/skills/{gemini,openai}/bin/ 아래 CLI 스크립트를
   서브프로세스로 호출한다. 각 CLI가 vault 서로게이트 방식으로 인증을 처리하므로
   API 키가 코드·인자·로그에 노출되지 않는다.
2. Streamlit Cloud: CLI가 없으므로 st.secrets에서 API 키를 읽어 REST API를 직접
   호출한다. (secrets에 GEMINI_API_KEY / OPENAI_API_KEY 등록 필요)

generate(provider_id, prompt, model) 하나만 호출하면 되며,
선택한 AI에 따라 호출 함수가 갈아끼워지는 구조다.
"""

import json
import os
import subprocess
import urllib.request

GEMINI_CLI = os.path.expanduser("~/workspace/skills/gemini/bin/gemini")
OPENAI_CLI = os.path.expanduser("~/workspace/skills/openai/bin/chatgpt")

PROVIDERS = {
    "gemini": {
        "label": "Gemini",
        # Prolific/벤치마크 리서치 기준 기본값: Pro 계열 (표준 LLM 중 최고 67%)
        # 무료 등급에서 할당량 오류(429)가 나면 gemini-3.6-flash로 변경
        "default_model": "gemini-pro-latest",
        "fallback_model": "gemini-3.6-flash",
        "cli": GEMINI_CLI,
        "secret_name": "GEMINI_API_KEY",
    },
    "openai": {
        "label": "OpenAI (ChatGPT)",
        "default_model": "gpt-4o-mini",
        "cli": OPENAI_CLI,
        "secret_name": "OPENAI_API_KEY",
    },
}


def get_secret(name: str) -> str:
    """st.secrets에서 값을 읽는다. Streamlit 밖에서는 빈 문자열."""
    try:
        import streamlit as st

        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def auth_status(provider_id: str) -> tuple[bool, str]:
    """현재 환경에서 해당 provider 호출 가능 여부와 설명을 반환."""
    cfg = PROVIDERS[provider_id]
    if os.path.exists(cfg["cli"]):
        return True, "로컬 CLI 인증 사용 가능"
    if get_secret(cfg["secret_name"]):
        return True, f"st.secrets의 {cfg['secret_name']} 사용"
    return False, f"인증 없음 — 로컬 CLI 또는 secrets의 {cfg['secret_name']} 필요"


def _call_cli(cli_path: str, prompt: str, model: str, timeout: int = 180) -> str:
    proc = subprocess.run(
        [cli_path, "--model", model],
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
    )
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()[:500]
        raise RuntimeError(f"CLI 호출 실패: {err or out[:500]}")
    if not out:
        raise RuntimeError("CLI가 빈 응답을 반환했습니다.")
    return out


def _call_gemini_rest(api_key: str, prompt: str, model: str, timeout: int = 180) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Gemini API 오류 (HTTP {e.code}): {e.read().decode('utf-8')[:300]}")
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini 응답 파싱 실패: {json.dumps(data, ensure_ascii=False)[:300]}")


def _call_openai_rest(api_key: str, prompt: str, model: str, timeout: int = 180) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenAI API 오류 (HTTP {e.code}): {e.read().decode('utf-8')[:300]}")
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"OpenAI 응답 파싱 실패: {json.dumps(data, ensure_ascii=False)[:300]}")


def generate(provider_id: str, prompt: str, model: str | None = None, timeout: int = 180) -> str:
    """선택된 AI provider로 텍스트를 생성한다. API 키 하드코딩 없음."""
    if provider_id not in PROVIDERS:
        raise ValueError(f"알 수 없는 provider: {provider_id}")
    cfg = PROVIDERS[provider_id]
    model = model or cfg["default_model"]

    if os.path.exists(cfg["cli"]):
        return _call_cli(cfg["cli"], prompt, model, timeout)

    api_key = get_secret(cfg["secret_name"])
    if not api_key:
        raise RuntimeError(
            f"API 키가 없습니다. Streamlit Cloud의 secrets에 "
            f"{cfg['secret_name']}를 등록하거나 로컬 CLI 환경을 사용하세요."
        )
    if provider_id == "gemini":
        return _call_gemini_rest(api_key, prompt, model, timeout)
    return _call_openai_rest(api_key, prompt, model, timeout)
