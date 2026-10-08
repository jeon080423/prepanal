"""AI provider 플러그인 레지스트리 (core/providers.py)

새 AI 추가 방법:
    1. AIProvider를 상속한 클래스 1개 작성 (_call_rest만 구현)
    2. _PROVIDER_CLASSES에 {"kind": 새클래스} 한 줄 추가
끝. 호출부에는 provider 이름에 대한 if/else가 없다.

두 가지 호출 경로:
1. 로컬: skills/*/bin CLI를 서브프로세스로 호출 (vault 서로게이트 인증 —
   API 키가 코드·인자·로그에 노출되지 않음)
2. Streamlit Cloud: st.secrets에서 API 키를 읽어 REST 직접 호출
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from config import PROVIDER_SPECS


class UnknownProviderError(ValueError):
    """등록되지 않은 provider를 요청했을 때 발생."""


class AIProvider(ABC):
    """새 AI는 이 클래스를 상속하고 _call_rest만 구현하면 된다."""

    provider_id: str = ""
    label: str = ""
    default_model: str = ""
    fallback_model: str = ""
    cli_path: str = ""
    secret_name: str = ""

    def __init__(self, provider_id: str, label: str, default_model: str,
                 fallback_model: str = "", cli: str = "",
                 secret_name: str = "") -> None:
        self.provider_id = provider_id
        self.label = label
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.cli_path = os.path.expanduser(cli)
        self.secret_name = secret_name

    # -- 인증 ---------------------------------------------------------------
    def is_available(self) -> bool:
        return bool(self.cli_path and os.path.exists(self.cli_path)) or bool(self._read_secret(self.secret_name))

    def auth_status(self) -> tuple[bool, str]:
        if self.cli_path and os.path.exists(self.cli_path):
            return True, "로컬 CLI 인증 사용 가능"
        if self._read_secret(self.secret_name):
            return True, f"st.secrets의 {self.secret_name} 사용"
        return False, f"인증 없음 — 로컬 CLI 또는 secrets의 {self.secret_name} 필요"

    @staticmethod
    def _read_secret(name: str) -> str:
        """st.secrets에서 값을 읽는다. Streamlit 밖에서는 빈 문자열."""
        try:
            import streamlit as st
            return str(st.secrets.get(name, "") or "")
        except Exception:
            return ""

    # -- 생성 (서브클래스는 _call_rest만 오버라이드) ---------------------------
    def generate(self, prompt: str, model: str | None = None,
                 timeout: int = 180) -> str:
        model = model or self.default_model
        if self.cli_path and os.path.exists(self.cli_path):
            return self._call_cli(prompt, model, timeout)
        api_key = self._read_secret(self.secret_name)
        if not api_key:
            raise RuntimeError(
                f"API 키가 없습니다. Streamlit Cloud의 secrets에 "
                f"{self.secret_name}를 등록하거나 로컬 CLI 환경을 사용하세요."
            )
        return self._call_rest(api_key, prompt, model, timeout)

    def _call_cli(self, prompt: str, model: str, timeout: int) -> str:
        proc = subprocess.run(
            [self.cli_path, "--model", model],
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

    @abstractmethod
    def _call_rest(self, api_key: str, prompt: str, model: str,
                   timeout: int) -> str:
        """REST 폴백. 각 provider의 응답 파싱만 구현하면 된다."""
        ...


class GeminiProvider(AIProvider):
    def _call_rest(self, api_key: str, prompt: str, model: str,
                   timeout: int = 180) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7},
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Gemini API 오류 (HTTP {e.code}): {e.read().decode('utf-8')[:300]}")
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError(
                f"Gemini 응답 파싱 실패: {json.dumps(data, ensure_ascii=False)[:300]}")


class OpenAIProvider(AIProvider):
    def _call_rest(self, api_key: str, prompt: str, model: str,
                   timeout: int = 180) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"OpenAI API 오류 (HTTP {e.code}): {e.read().decode('utf-8')[:300]}")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise RuntimeError(
                f"OpenAI 응답 파싱 실패: {json.dumps(data, ensure_ascii=False)[:300]}")


# ----------------------------------------------------------------------------
# 레지스트리
# ----------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, AIProvider] = {}

# 새 provider 종류 등록은 여기 한 줄 (kind → 클래스)
_PROVIDER_CLASSES: dict[str, type[AIProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
}


def register_provider(provider_id: str, provider: AIProvider) -> AIProvider:
    """실행 중 provider 등록/교체. 테스트용 더미 provider 주입에도 쓴다."""
    PROVIDER_REGISTRY[provider_id] = provider
    return provider


def get_provider(provider_id: str) -> AIProvider:
    try:
        return PROVIDER_REGISTRY[provider_id]
    except KeyError:
        raise UnknownProviderError(
            f"알 수 없는 AI provider: '{provider_id}'. "
            f"등록된 provider: {sorted(PROVIDER_REGISTRY)}"
        ) from None


def _build_default_registry() -> None:
    for pid, spec in PROVIDER_SPECS.items():
        cls = _PROVIDER_CLASSES[spec["kind"]]
        register_provider(pid, cls(
            provider_id=pid,
            label=spec["label"],
            default_model=spec["default_model"],
            fallback_model=spec.get("fallback_model", ""),
            cli=spec.get("cli", ""),
            secret_name=spec.get("secret_name", ""),
        ))


_build_default_registry()


# -- 하위 호환 진입점 (호출부는 레지스트리로 위임, 이름 분기 없음) -------------
def generate(provider_id: str, prompt: str, model: str | None = None,
             timeout: int = 180) -> str:
    """선택된 AI provider로 텍스트 생성. API 키 하드코딩 없음."""
    return get_provider(provider_id).generate(prompt, model, timeout)


def auth_status(provider_id: str) -> tuple[bool, str]:
    """현재 환경에서 해당 provider 호출 가능 여부와 설명."""
    return get_provider(provider_id).auth_status()
