"""Bundle-3 워커 B 검증 테스트 (providers.py: Gemini thinking + 키 라운드로빈).

실제 API 호출 없음 — urllib.request.urlopen을 mock하고, generate() 테스트는
_call_rest를 스텁으로 교체한다. API 키는 전부 가짜 문자열이다.
실행: ~/workspace/.venv-ahptest/bin/python -m pytest tests/test_providers_bundle3.py -v
"""

import json

import pytest

from core.providers import GeminiProvider, _is_retryable_error


FAKE_KEYS = ["fake-key-1", "fake-key-2", "fake-key-3"]


def make_gemini():
    """REST 경로를 타는 GeminiProvider (cli 없음 → CLI 분기 우회)."""
    p = GeminiProvider(
        provider_id="gemini-test",
        label="Gemini(test)",
        default_model="gemini-3.5-flash-lite",
        fallback_model="gemini-3.8-flash",
        cli="",
        secret_name="GEMINI_API_KEY",
        models=[
            {"id": "gemini-3.5-flash-lite"},
            {"id": "gemini-3.6-flash"},
            {"id": "gemini-3.8-flash"},
        ],
    )
    p._effective_keys = lambda: list(FAKE_KEYS)  # 가짜 키만 사용
    return p


class FakeResp:
    """urlopen() mock 응답 (context manager)."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def ok_payload(text='{"a": 1}', finish="STOP", usage=None):
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": text}]}, "finishReason": finish}
        ],
        "usageMetadata": usage or {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30,
        },
    }
    return payload


def run_call_rest(monkeypatch, payload):
    """_call_rest 1회 호출을 mock urlopen으로 실행하고 (결과, 요청 body)를 반환."""
    provider = make_gemini()
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        return FakeResp(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = provider._call_rest("fake-key-1", "prompt", "gemini-3.5-flash-lite", 30, 0.7)
    return result, captured


# ---------------------------------------------------------------------------
# 테스트 1: thinking / 빈 응답 처리
# ---------------------------------------------------------------------------

def test_generation_config_has_thinking_and_max_tokens(monkeypatch):
    """generationConfig에 thinkingLevel=LOW, maxOutputTokens=8192, temperature 유지."""
    _, captured = run_call_rest(monkeypatch, ok_payload())
    cfg = captured["body"]["generationConfig"]
    assert cfg["temperature"] == 0.7
    assert cfg["maxOutputTokens"] == 8192
    assert cfg["thinkingConfig"] == {"thinkingLevel": "LOW"}
    # thinkingBudget(2.5 방식)과 함께 보내지 않음 — 상호 배타적이라 400 원인
    assert "thinkingBudget" not in cfg["thinkingConfig"]


def test_normal_response_returns_text(monkeypatch):
    result, _ = run_call_rest(monkeypatch, ok_payload('{"x": 1}'))
    assert result == '{"x": 1}'


def test_max_tokens_raises_retryable(monkeypatch):
    """finishReason=MAX_TOKENS → 재시도 가능 오류 (폴백 판별에 걸림)."""
    provider = make_gemini()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: FakeResp(ok_payload('{"partial"', finish="MAX_TOKENS")),
    )
    with pytest.raises(RuntimeError) as exc:
        provider._call_rest("fake-key-1", "p", "gemini-3.5-flash-lite", 30)
    assert "MAX_TOKENS" in str(exc.value)
    assert _is_retryable_error(str(exc.value))


def test_empty_candidates_raises_retryable(monkeypatch):
    """candidates가 비어 있음 → 재시도 가능 오류."""
    provider = make_gemini()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: FakeResp({"candidates": []}),
    )
    with pytest.raises(RuntimeError) as exc:
        provider._call_rest("fake-key-1", "p", "gemini-3.5-flash-lite", 30)
    assert _is_retryable_error(str(exc.value))


def test_empty_parts_raises_retryable(monkeypatch):
    """content/parts에 텍스트가 없음 → 재시도 가능 오류."""
    provider = make_gemini()
    payload = {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: FakeResp(payload),
    )
    with pytest.raises(RuntimeError) as exc:
        provider._call_rest("fake-key-1", "p", "gemini-3.5-flash-lite", 30)
    assert _is_retryable_error(str(exc.value))


def test_usage_metadata_token_cap_raises_retryable(monkeypatch):
    """candidatesTokenCount가 maxOutputTokens에 도달 → 잘림으로 판정."""
    provider = make_gemini()
    payload = ok_payload('{"x": 1}', usage={"candidatesTokenCount": 8192})
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: FakeResp(payload),
    )
    with pytest.raises(RuntimeError) as exc:
        provider._call_rest("fake-key-1", "p", "gemini-3.5-flash-lite", 30)
    assert "truncated" in str(exc.value).lower()
    assert _is_retryable_error(str(exc.value))


def test_thought_parts_filtered(monkeypatch):
    """thought 플래그 part는 본문에서 제외하고 텍스트 part만 반환."""
    provider = make_gemini()
    payload = {
        "candidates": [{
            "content": {"parts": [
                {"text": "생각 중...", "thought": True},
                {"text": '{"real": true}'},
            ]},
            "finishReason": "STOP",
        }]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: FakeResp(payload),
    )
    assert provider._call_rest("fake-key-1", "p", "gemini-3.5-flash-lite", 30) == '{"real": true}'


def test_max_tokens_falls_back_to_next_key(monkeypatch):
    """generate() 경유: 키1에서 MAX_TOKENS → 키2로 폴백해 성공."""
    provider = make_gemini()
    calls = []

    def stub(api_key, prompt, model, timeout, temperature=0.7):
        calls.append(api_key)
        if api_key == "fake-key-1":
            raise RuntimeError("Gemini output truncated: finishReason=MAX_TOKENS (max_tokens)")
        return '{"ok": true}'

    monkeypatch.setattr(provider, "_call_rest", stub)
    assert provider.generate("p") == '{"ok": true}'
    assert calls[0] == "fake-key-1"
    assert calls[-1] == "fake-key-2"


# ---------------------------------------------------------------------------
# 테스트 2: 라운드로빈
# ---------------------------------------------------------------------------

def test_round_robin_start_key_rotates(monkeypatch):
    """연속 generate() 호출의 시작 키가 0→1→2→0→1→2로 순환."""
    provider = make_gemini()
    starts = []

    def stub(api_key, prompt, model, timeout, temperature=0.7):
        starts.append(provider._last_start_key_index)
        return f"ok-{api_key}"

    monkeypatch.setattr(provider, "_call_rest", stub)
    results = [provider.generate("p") for _ in range(6)]
    assert starts == [0, 1, 2, 0, 1, 2]
    assert results == [
        "ok-fake-key-1", "ok-fake-key-2", "ok-fake-key-3",
        "ok-fake-key-1", "ok-fake-key-2", "ok-fake-key-3",
    ]


def test_round_robin_single_key_is_noop(monkeypatch):
    """키가 1개면 시작 키는 항상 0 (회전 없음, 오류 없음)."""
    provider = make_gemini()
    provider._effective_keys = lambda: ["only-fake-key"]
    monkeypatch.setattr(provider, "_call_rest",
                        lambda *a, **k: "ok")
    for _ in range(3):
        assert provider.generate("p") == "ok"
        assert provider._last_start_key_index == 0


# ---------------------------------------------------------------------------
# 테스트 3: 429 폴백 유지
# ---------------------------------------------------------------------------

def test_429_falls_back_to_next_key(monkeypatch):
    """첫 키에서 429 → 다음 키로 넘어가 성공. 호출 순서가 시작 키를 따른다."""
    provider = make_gemini()
    calls = []

    def stub(api_key, prompt, model, timeout, temperature=0.7):
        calls.append(api_key)
        if api_key == "fake-key-1":
            raise RuntimeError("Gemini API 오류 (HTTP 429): quota exceeded")
        return "ok"

    monkeypatch.setattr(provider, "_call_rest", stub)
    assert provider.generate("p") == "ok"
    # 키1(429) → 키1의 다음 모델 2개도 429이므로 모델 순회 후 키2로 이동
    assert calls[0] == "fake-key-1"
    assert calls[-1] == "fake-key-2"


def test_429_fallback_from_rotated_start(monkeypatch):
    """라운드로빈 시작 키가 1인 상태에서 429 → 키3(인덱스 2)으로 폴백."""
    provider = make_gemini()
    provider._key_cursor = 1  # 다음 호출 시작 키 = 1
    calls = []

    def stub(api_key, prompt, model, timeout, temperature=0.7):
        calls.append(api_key)
        if api_key == "fake-key-2":
            raise RuntimeError("Gemini API 오류 (HTTP 429): quota exceeded")
        return "ok"

    monkeypatch.setattr(provider, "_call_rest", stub)
    assert provider.generate("p") == "ok"
    assert provider._last_start_key_index == 1
    assert calls[0] == "fake-key-2"
    assert calls[-1] == "fake-key-3"


def test_non_retryable_error_raises_immediately(monkeypatch):
    """401 같은 비재시도 오류는 다음 키로 넘어가지 않고 즉시 전파."""
    provider = make_gemini()
    calls = []

    def stub(api_key, prompt, model, timeout, temperature=0.7):
        calls.append(api_key)
        raise RuntimeError("Gemini API 오류 (HTTP 401): API_KEY_INVALID")

    monkeypatch.setattr(provider, "_call_rest", stub)
    with pytest.raises(RuntimeError, match="401"):
        provider.generate("p")
    assert len(calls) == 1


def test_all_keys_fail_raises_summary(monkeypatch):
    """전 키·전 모델 실패 시 요약 오류 (기존 동작 유지)."""
    provider = make_gemini()
    monkeypatch.setattr(
        provider, "_call_rest",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("Gemini API 오류 (HTTP 503): overloaded")),
    )
    with pytest.raises(RuntimeError, match="모든 API 키"):
        provider.generate("p")
