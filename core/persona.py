"""페르소나 샘플링 (core/persona.py)

인구통계 분포 → 가상 응답자 프로필 생성.
고정밀 모드용 인터뷰 트랜스크립트 기반 텍스트도 여기서 만든다.
"""

import random

from config import TRANSCRIPT_EXCERPT_CHARS


def weighted_choice(dist: dict[str, float], rng: random.Random) -> str:
    total = sum(dist.values())
    if total <= 0:
        return next(iter(dist))
    r = rng.uniform(0, total)
    upto = 0.0
    for k, v in dist.items():
        upto += v
        if r <= upto:
            return k
    return next(reversed(dist))


def sample_persona(persona_dims: dict[str, dict[str, float]],
                   rng: random.Random) -> dict[str, str]:
    """차원별 분포에서 가중 샘플링해 1명의 페르소나를 만든다."""
    return {dim: weighted_choice(dist, rng) for dim, dist in persona_dims.items()}


def persona_to_text(persona: dict[str, str]) -> str:
    return ", ".join(f"{dim} {val}" for dim, val in persona.items())


def build_persona_texts(personas: list[dict], rng: random.Random,
                        transcripts: list[dict] | None = None) -> list[str]:
    """응답자별 프로필 텍스트.

    transcripts가 주어지면 고정밀 모드(스탠포드 Generative Agents 방식):
    인터뷰 발췌문을 페르소나 컨텍스트로 사용한다.
    """
    texts = []
    for p in personas:
        if transcripts:
            t = rng.choice(transcripts)
            excerpt = t["text"][:TRANSCRIPT_EXCERPT_CHARS]
            texts.append(
                f"인터뷰 기반 페르소나 '{t['name']}' — 다음 실제 인터뷰 내용을 반영:\n{excerpt}"
            )
        else:
            texts.append(persona_to_text(p))
    return texts
