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
import socket
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
    models: list = []

    def __init__(self, provider_id: str, label: str, default_model: str,
                 fallback_model: str = "", cli: str = "",
                 secret_name: str = "", models: list = None) -> None:
        self.provider_id = provider_id
        self.label = label
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.cli_path = os.path.expanduser(cli)
        self.secret_name = secret_name
        self.models = models or []
        self._custom_key = ""  # 사용자가 직접 입력한 API 키 (세션 한정)
        self.last_used_model = ""  # 마지막 생성에 실제 사용된 모델 (자동 전환 추적용)

    # -- 사용자 API 키 ------------------------------------------------------
    def set_custom_key(self, api_key: str) -> None:
        """사용자가 직접 입력한 API 키를 설정 (세션 메모리에만 보관)."""
        self._custom_key = (api_key or "").strip()

    def get_custom_key(self) -> str:
        return self._custom_key

    def has_custom_key(self) -> bool:
        return bool(self._custom_key)

    def clear_custom_key(self) -> None:
        self._custom_key = ""

    def test_key(self, api_key: str, timeout: int = 30) -> tuple[bool, str]:
        """API 키 유효성을 사전 테스트. (실제 생성 전 가벼운 호출)

        Returns: (성공 여부, 메시지)
        """
        api_key = (api_key or "").strip()
        if not api_key:
            return False, "API 키를 입력하세요."
        try:
            # 최소 토큰으로 가벼운 테스트 호출
            self._call_rest(api_key, "Say OK.", self.default_model, timeout)
            return True, "API 키가 유효합니다. ✅"
        except RuntimeError as e:
            err = str(e)
            if "401" in err or "403" in err or "API_KEY_INVALID" in err or "invalid" in err.lower():
                return False, "❌ 유효하지 않은 API 키입니다. 키를 확인하세요."
            if "429" in err:
                return False, "❌ 할당량 초과 (429). 잠시 후 다시 시도하세요."
            return False, f"❌ 연결 실패: {err[:200]}"
        except Exception as e:
            return False, f"❌ 오류: {str(e)[:200]}"

    # -- 인증 ---------------------------------------------------------------
    def _effective_keys(self) -> list[str]:
        """사용 가능한 API 키 목록. 사용자 입력 키 > st.secrets 순서.

        st.secrets의 값은 단일 문자열 또는 리스트 모두 지원한다.
        """
        if self._custom_key:
            return [self._custom_key]
        return self._read_secret_keys(self.secret_name)

    def _effective_key(self) -> str:
        """하위 호환: 첫 번째 키 반환."""
        keys = self._effective_keys()
        return keys[0] if keys else ""

    def is_available(self) -> bool:
        return bool(self.cli_path and os.path.exists(self.cli_path)) or bool(self._effective_keys())

    def auth_status(self) -> tuple[bool, str]:
        if self.cli_path and os.path.exists(self.cli_path):
            return True, "로컬 CLI 인증 사용 가능"
        if self._custom_key:
            return True, "사용자 입력 API 키 사용 중"
        keys = self._read_secret_keys(self.secret_name)
        if keys:
            return True, f"st.secrets의 {self.secret_name} 사용 ({len(keys)}개 키)"
        return False, f"인증 없음 — API 키를 직접 입력하거나 secrets에 {self.secret_name} 등록 필요"

    @staticmethod
    def _read_secret_keys(name: str) -> list[str]:
        """st.secrets에서 키 목록을 읽는다. 단일 문자열 또는 리스트 지원.
        Streamlit 밖에서는 빈 리스트."""
        try:
            import streamlit as st
            val = st.secrets.get(name, "")
            if isinstance(val, (list, tuple)):
                return [str(v).strip() for v in val if str(v).strip()]
            val = str(val or "").strip()
            return [val] if val else []
        except Exception:
            return []

    @staticmethod
    def _read_secret(name: str) -> str:
        """하위 호환: 첫 번째 키 반환."""
        keys = AIProvider._read_secret_keys(name)
        return keys[0] if keys else ""

    # -- 생성 (서브클래스는 _call_rest만 오버라이드) ---------------------------
    def _candidate_models(self, requested: str) -> list[str]:
        """시도할 모델 목록: 요청 모델 먼저, 그 다음 models 목록 순서대로.

        토큰 소진(429)이나 과부하(503) 시 자동으로 다음 모델로 넘어간다.
        """
        seen: list[str] = []
        if requested:
            seen.append(requested)
        for m in self.models:
            mid = m["id"] if isinstance(m, dict) else m
            if mid not in seen:
                seen.append(mid)
        if self.fallback_model and self.fallback_model not in seen:
            seen.append(self.fallback_model)
        return seen

    def generate(self, prompt: str, model: str | None = None,
                 timeout: int = 180) -> str:
        model = model or self.default_model
        if self.cli_path and os.path.exists(self.cli_path):
            self.last_used_model = model
            return self._call_cli(prompt, model, timeout)
        api_keys = self._effective_keys()
        if not api_keys:
            raise RuntimeError(
                f"API 키가 없습니다. 사이드바에서 API 키를 직접 입력하거나, "
                f"Streamlit Cloud의 secrets에 {self.secret_name}를 등록하세요."
            )
        candidates = self._candidate_models(model)
        last_err: Exception | None = None
        # 키 로테이션: 429가 나면 다음 키로 전환
        for key_idx, api_key in enumerate(api_keys):
            for cand in candidates:
                try:
                    result = self._call_rest(api_key, prompt, cand, timeout)
                    self.last_used_model = cand
                    if len(api_keys) > 1:
                        self.last_used_model += f" (키 {key_idx + 1}/{len(api_keys)})"
                    return result
                except RuntimeError as e:
                    err = str(e).lower()
                    # 429(할당량 초과)/503(과부하)/404(모델 미지원)/타임아웃이면 다음으로 전환
                    if ("429" not in err and "503" not in err and "404" not in err
                            and "timed out" not in err and "timeout" not in err
                            and "no longer available" not in err):
                        raise
                    last_err = e
                    continue
        raise RuntimeError(
            f"모든 API 키({len(api_keys)}개)와 모델 시도 실패. "
            f"잠시 후 다시 시도하세요. 마지막 오류: {last_err}"
        )

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
        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
            raise RuntimeError(f"The read operation timed out: {e}")
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RuntimeError(
                f"Gemini 응답 파싱 실패: {json.dumps(data, ensure_ascii=False)[:300]}")


# ----------------------------------------------------------------------------
# 레지스트리
# ----------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, AIProvider] = {}

# 새 provider 종류 등록은 여기 한 줄 (kind → 클래스)
_PROVIDER_CLASSES: dict[str, type[AIProvider]] = {
    "gemini": GeminiProvider,
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
            models=spec.get("models", []),
        ))


_build_default_registry()


# -- 하위 호환 진입점 (호출부는 레지스트리로 위임, 이름 분기 없음) -------------
def generate(provider_id: str, prompt: str, model: str | None = None,
             timeout: int = 180) -> str:
    """선택된 AI provider로 텍스트 생성. API 키 하드코딩 없음.

    요청한 provider가 할당량 초과(429)/과부하(503)/타임아웃으로 실패하면
    자동으로 다른 provider로 전환한다. 인증 오류(API 키 없음 등)는 전환하지 않는다.
    """
    provider = get_provider(provider_id)
    try:
        return provider.generate(prompt, model, timeout)
    except RuntimeError as e:
        err = str(e).lower()
        # 전환 대상 오류가 아니면 그대로 전파
        if ("429" not in err and "503" not in err and "404" not in err
                and "timed out" not in err and "timeout" not in err
                and "no longer available" not in err):
            raise
        # 다른 provider로 폴백
        for other_id in PROVIDER_REGISTRY:
            if other_id == provider_id:
                continue
            try:
                other = get_provider(other_id)
                result = other.generate(prompt, None, timeout)
                # 실제 사용된 provider 추적
                provider.last_used_model = (
                    f"{other.label} ({other.last_used_model}) [자동 전환]"
                )
                return result
            except RuntimeError:
                continue
        raise RuntimeError(
            f"모든 AI provider 시도 실패. 잠시 후 다시 시도하세요. "
            f"마지막 오류: {e}"
        )


def auth_status(provider_id: str) -> tuple[bool, str]:
    """현재 환경에서 해당 provider 호출 가능 여부와 설명."""
    return get_provider(provider_id).auth_status()
