"""저장소 추상화 (core/store.py)

세션 상태 접근을 Store 인터페이스 뒤에 둔다.
지금은 SessionStore(Streamlit 세션). 나중에 DB 기반으로 바꿔도
호출부 코드는 손대지 않는다.

사용법:
    store = SessionStore()          # Streamlit 앱에서
    store = MemoryStore()           # 테스트·향후 REST API에서
    store.set_default("step", 1)
    step = store.get("step")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Store(ABC):
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def __contains__(self, key: object) -> bool:
        ...

    def set_default(self, key: str, value: Any) -> Any:
        """없을 때만 기본값을 넣고 현재 값을 반환."""
        if key not in self:
            self.set(key, value)
        return self.get(key)


class MemoryStore(Store):
    """dict 기반. 테스트·비Streamlit 환경용."""

    def __init__(self, initial: dict | None = None) -> None:
        self._data: dict = dict(initial or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def __contains__(self, key: object) -> bool:
        return key in self._data


class SessionStore(Store):
    """st.session_state 래퍼. streamlit import는 여기서만 지연 수행."""

    def _state(self):
        import streamlit as st  # noqa: PLC0415 (UI 레이어에서만 필요)
        return st.session_state

    def get(self, key: str, default: Any = None) -> Any:
        state = self._state()
        return state[key] if key in state else default

    def set(self, key: str, value: Any) -> None:
        self._state()[key] = value

    def delete(self, key: str) -> None:
        state = self._state()
        if key in state:
            del state[key]

    def __contains__(self, key: object) -> bool:
        return key in self._state()
