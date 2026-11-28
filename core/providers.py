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


# 재시도 가능한(다음 키·모델·provider로 폴백을 타는) 오류 판별 마커.
# 기존 마커(429/503/404/타임아웃)에 더해, 잘린·빈 응답도 폴백 대상이다.
# Gemini _call_rest가 던지는 MAX_TOKENS/빈 응답 오류는 이 마커를 포함한다.
_RETRYABLE_ERROR_MARKERS = (
    "429", "503", "404", "timed out", "timeout",
    "no longer available",
    "max_tokens", "empty response", "truncated",
)


def _is_retryable_error(err: str) -> bool:

    """다음 키·모델·provider로 넘어가도 되는 오류인지 판별한다."""

    lowered = err.lower()

    return any(m in lowered for m in _RETRYABLE_ERROR_MARKERS)


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

        self._key_cursor = 0  # 라운드로빈: 다음 generate()가 시작할 키 인덱스

        self._last_start_key_index = -1  # 마지막 generate()의 시작 키 인덱스 (관찰용)

        self._last_fallback_used = False  # 마지막 generate()에서 키 폴백(다음 키로 전환)이 있었는지 (관찰용)

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
                 timeout: int = 180, temperature: float = 0.7) -> str:

        """텍스트 생성. temperature는 선택적 인자 (기본값 0.7 = 기존 동작 유지).

        REST 경로(Gemini/Groq)에서는 generationConfig/body에 그대로 실린다.

        로컬 CLI 경로는 CLI가 temperature를 하드코딩(0.7)하고 있어 이 인자를

        무시한다 — CLI 경유 온도 실험은 benchmarks/run_temperature.py의

        REST 직접 호출 방식을 쓴다.

        """

        model = model or self.default_model

        # 실행 로그용 관찰 상태 초기화: 호출 시작 시마다 False로 되돌린다.
        # CLI 경로로 바로 반환되는 경우에도 폴백 없음(False)이 보고되며,
        # 시작 인덱스도 -1로 되돌려 이전 호출의 값이 남지 않게 한다.
        self._last_fallback_used = False

        self._last_start_key_index = -1

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

        # 키 라운드로빈: 매 호출마다 시작 키를 한 칸씩 순환시켜 여러 키에
        # 부하를 분산시킨다. 호출 1회마다 시작 키가 바뀌므로, 첫 키만 계속
        # 소진되는 기존 동작이 사라진다. 429 등이 나면 기존처럼 다음 키로 전환.
        n_keys = len(api_keys)

        start_key = self._key_cursor % n_keys

        self._last_start_key_index = start_key

        self._key_cursor += 1

        ordered_keys = [(start_key + i) % n_keys for i in range(n_keys)]

        for key_pos, key_idx in enumerate(ordered_keys):

            # key_pos > 0에 도달했다는 것은 이전 키의 모든 후보 모델이
            # 재시도 가능 오류로 실패해 다음 키로 넘어왔다는 뜻이다.
            # 같은 키 안에서의 모델 전환은 키 폴백이 아니므로 False를 유지한다.
            if key_pos > 0:

                self._last_fallback_used = True

            api_key = api_keys[key_idx]

            for cand in candidates:

                try:

                    result = self._call_rest(api_key, prompt, cand, timeout,
                                             temperature)

                    self.last_used_model = cand

                    if len(api_keys) > 1:

                        self.last_used_model += f" (키 {key_idx + 1}/{len(api_keys)})"

                    return result

                except RuntimeError as e:

                    err = str(e).lower()

                    # 429(할당량 초과)/503(과부하)/404(모델 미지원)/타임아웃/
                    # 잘린·빈 응답이면 다음 키·모델로 전환

                    if not _is_retryable_error(err):

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
                   timeout: int, temperature: float = 0.7) -> str:

        """REST 폴백. 각 provider의 응답 파싱만 구현하면 된다."""

        ...


class GeminiProvider(AIProvider):

    # Gemini 3.x thinking 설정. 3.x는 thinkingBudget(2.5 방식)이 아니라
    # thinkingLevel을 쓴다(둘은 상호 배타적 — 함께 보내면 400 오류).
    # "MINIMAL"은 gemini-3.8-flash 등 일부 모델에서 400을 내므로, 전 모델이
    # 받는 최소 수준인 "LOW"를 쓴다(3.5-flash-lite/3.6-flash/3.8-flash 공통).
    # thinking 토큰은 maxOutputTokens 안에 counted되므로, LOW로 묶어 두면
    # 출력 토큰이 thinking에 잠식돼 분포 JSON이 잘리는 일을 줄인다.
    GEMINI_THINKING_LEVEL = "LOW"

    # 분포 JSON 출력 여유분. thinking 포함 출력 상한.
    GEMINI_MAX_OUTPUT_TOKENS = 8192

    def _call_rest(self, api_key: str, prompt: str, model: str,
                   timeout: int = 180, temperature: float = 0.7) -> str:

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        body = json.dumps({

            "contents": [{"parts": [{"text": prompt}]}],

            "generationConfig": {

                "temperature": temperature,

                "maxOutputTokens": self.GEMINI_MAX_OUTPUT_TOKENS,

                "thinkingConfig": {"thinkingLevel": self.GEMINI_THINKING_LEVEL},

            },

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

        # 응답 검증: finishReason·usageMetadata·본문 텍스트를 검사한다.
        # MAX_TOKENS로 잘렸거나 본문이 비어 있으면, 잘린 JSON을 그대로
        # 돌려주지 않고 재시도 가능한 오류를 던져 다음 키·모델 폴백을 탄다.
        # 오류 메시지에는 _is_retryable_error 마커(max_tokens/empty response/
        # truncated)를 포함시켜 generate()의 폴백 판별에 걸리게 한다.
        try:

            candidates = data.get("candidates") or []

            if not candidates:

                raise RuntimeError(
                    "Gemini empty response: candidates가 비어 있습니다. "
                    f"(empty response, 응답: {json.dumps(data, ensure_ascii=False)[:300]})")

            cand = candidates[0]

            finish_reason = cand.get("finishReason", "")

            usage = data.get("usageMetadata") or {}

            if finish_reason == "MAX_TOKENS":

                raise RuntimeError(
                    "Gemini output truncated: finishReason=MAX_TOKENS — "
                    "출력 토큰 한도에 걸려 응답이 잘렸습니다. "
                    f"(max_tokens, usageMetadata={json.dumps(usage, ensure_ascii=False)[:200]})")

            cand_tokens = usage.get("candidatesTokenCount")

            if (cand_tokens is not None
                    and cand_tokens >= self.GEMINI_MAX_OUTPUT_TOKENS):

                raise RuntimeError(
                    "Gemini output truncated: candidatesTokenCount가 "
                    "maxOutputTokens에 도달했습니다. "
                    f"(max_tokens, usageMetadata={json.dumps(usage, ensure_ascii=False)[:200]})")

            parts = (cand.get("content") or {}).get("parts") or []

            text = "".join(
                p.get("text", "")
                for p in parts
                if isinstance(p, dict) and "text" in p and not p.get("thought")
            )

            if not text.strip():

                raise RuntimeError(
                    "Gemini empty response: 텍스트 part가 비어 있습니다. "
                    f"(empty response, finishReason={finish_reason}, "
                    f"usageMetadata={json.dumps(usage, ensure_ascii=False)[:200]})")

            return text

        except RuntimeError:

            raise

        except (KeyError, IndexError, TypeError, AttributeError):

            raise RuntimeError(

                f"Gemini 응답 파싱 실패: {json.dumps(data, ensure_ascii=False)[:300]}")


class GroqProvider(AIProvider):

    """Groq API (OpenAI 호환). 무료 티어 제공."""

    def _call_rest(self, api_key: str, prompt: str, model: str,
                   timeout: int = 180, temperature: float = 0.7) -> str:

        url = "https://api.groq.com/openai/v1/chat/completions"

        body = json.dumps({

            "model": model,

            "messages": [{"role": "user", "content": prompt}],

            "temperature": temperature,

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

                f"Groq API 오류 (HTTP {e.code}): {e.read().decode('utf-8')[:300]}")

        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:

            raise RuntimeError(f"The read operation timed out: {e}")

        try:

            return data["choices"][0]["message"]["content"]

        except (KeyError, IndexError):

            raise RuntimeError(

                f"Groq 응답 파싱 실패: {json.dumps(data, ensure_ascii=False)[:300]}")


class CloudflareProvider(AIProvider):

    """Cloudflare Workers AI (OpenAI 호환 엔드포인트). 무료 티어 제공.

    account_id는 URL 경로에 들어가므로 secrets의 CLOUDFLARE_ACCOUNT_ID에서
    읽는다. _build_default_registry는 extra 필드를 넘기지 않으므로,
    _call_rest 안에서 _read_secret으로 직접 읽는 방식을 쓴다.
    API 토큰(CLOUDFLARE_API_TOKEN)은 베이스 클래스의 _effective_keys 경유로
    api_key 인자로 들어온다. 비밀값은 코드에 하드코딩하지 않는다.
    """

    def _call_rest(self, api_key: str, prompt: str, model: str,
                   timeout: int = 180, temperature: float = 0.7) -> str:

        account_id = self._read_secret("CLOUDFLARE_ACCOUNT_ID")

        if not account_id:

            raise RuntimeError(
                "Cloudflare account_id가 없습니다. "
                "Streamlit Cloud의 secrets에 CLOUDFLARE_ACCOUNT_ID를 등록하세요."
            )

        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/v1/chat/completions"
        )

        body = json.dumps({

            "model": model,

            "messages": [{"role": "user", "content": prompt}],

            "temperature": temperature,

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

                f"Cloudflare API 오류 (HTTP {e.code}): {e.read().decode('utf-8')[:300]}")

        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:

            raise RuntimeError(f"The read operation timed out: {e}")

        try:

            return data["choices"][0]["message"]["content"]

        except (KeyError, IndexError):

            raise RuntimeError(

                f"Cloudflare 응답 파싱 실패: {json.dumps(data, ensure_ascii=False)[:300]}")


class SambaNovaProvider(AIProvider):

    """SambaNova Cloud API (OpenAI 호환). 무료 티어 제공."""

    def _call_rest(self, api_key: str, prompt: str, model: str,
                   timeout: int = 180, temperature: float = 0.7) -> str:

        url = "https://api.sambanova.ai/v1/chat/completions"

        body = json.dumps({

            "model": model,

            "messages": [{"role": "user", "content": prompt}],

            "temperature": temperature,

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

                f"SambaNova API 오류 (HTTP {e.code}): {e.read().decode('utf-8')[:300]}")

        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:

            raise RuntimeError(f"The read operation timed out: {e}")

        try:

            return data["choices"][0]["message"]["content"]

        except (KeyError, IndexError):

            raise RuntimeError(

                f"SambaNova 응답 파싱 실패: {json.dumps(data, ensure_ascii=False)[:300]}")


# ----------------------------------------------------------------------------

# 레지스트리

# ----------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, AIProvider] = {}


# 새 provider 종류 등록은 여기 한 줄 (kind → 클래스)

_PROVIDER_CLASSES: dict[str, type[AIProvider]] = {

    "gemini": GeminiProvider,

    "groq": GroqProvider,

    "cloudflare": CloudflareProvider,

    "sambanova": SambaNovaProvider,

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


# -- 마지막 호출 실행 정보 (워커 C가 services.py에서 읽어 result["execution"]에 남김) --

LAST_CALL_INFO: dict = {"start_key_index": None, "fallback_used": False}


def get_last_call_info() -> dict:

    """마지막 generate() 호출의 실행 정보를 복사본으로 반환한다."""

    return dict(LAST_CALL_INFO)


def _update_last_call_info(provider) -> None:

    """provider 인스턴스의 관찰 속성을 읽어 LAST_CALL_INFO를 갱신한다.

    해당 속성이 없는 provider(구형/더미 인스턴스)에도 getattr으로 방어한다.

    _last_start_key_index == -1은 '키 기반 호출이 아님'(CLI 경로 등)을 뜻하므로

    None으로 정규화한다.

    """

    start = getattr(provider, "_last_start_key_index", None)

    if isinstance(start, int) and start < 0:

        start = None

    LAST_CALL_INFO["start_key_index"] = start

    LAST_CALL_INFO["fallback_used"] = bool(

        getattr(provider, "_last_fallback_used", False))


# -- 하위 호환 진입점 (호출부는 레지스트리로 위임, 이름 분기 없음) -------------

def generate(provider_id: str, prompt: str, model: str | None = None,
             timeout: int = 180, temperature: float = 0.7) -> str:

    """선택된 AI provider로 텍스트 생성. API 키 하드코딩 없음.

    temperature는 선택적 인자 (기본값 0.7 = 기존 동작 유지). 로컬 CLI 경로는

    CLI가 temperature를 하드코딩하고 있어 무시된다.

    요청한 provider가 할당량 초과(429)/과부하(503)/타임아웃으로 실패하면

    자동으로 다른 provider로 전환한다. 인증 오류(API 키 없음 등)는 전환하지 않는다.

    """

    provider = get_provider(provider_id)

    try:

        result = provider.generate(prompt, model, timeout, temperature)

        _update_last_call_info(provider)

        return result

    except RuntimeError as e:

        err = str(e).lower()

        # 전환 대상 오류가 아니면 그대로 전파 (실행 정보는 남기고 전파)

        if not _is_retryable_error(err):

            _update_last_call_info(provider)

            raise

        # 다른 provider로 폴백

        for other_id in PROVIDER_REGISTRY:

            if other_id == provider_id:

                continue

            try:

                other = get_provider(other_id)

                result = other.generate(prompt, None, timeout, temperature)

                # 실제 사용된 provider 추적 (더미 provider에는
                # last_used_model이 없을 수 있어 getattr으로 방어한다)

                provider.last_used_model = (

                    f"{other.label} ({getattr(other, 'last_used_model', '')}) [자동 전환]"

                )

                # 실제로 생성에 성공한 provider의 실행 정보를 남긴다.

                _update_last_call_info(other)

                return result

            except RuntimeError:

                continue

        _update_last_call_info(provider)

        raise RuntimeError(

            f"모든 AI provider 시도 실패. 잠시 후 다시 시도하세요. "

            f"마지막 오류: {e}"

        )


def auth_status(provider_id: str) -> tuple[bool, str]:

    """현재 환경에서 해당 provider 호출 가능 여부와 설명."""

    return get_provider(provider_id).auth_status()
