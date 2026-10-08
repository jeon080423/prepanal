"""가상응답자 스튜디오 (Synthetic Respondent Studio)

설문 설계 → 페르소나 설계 → 분포 학습 → 가상 응답 생성 → 결과
5단계 마법사형 Streamlit 앱.

방법론 (정직한 고지):
- GPU 파인튜닝이 아님. 분포 학습(경험분포 추출) + 페르소나 시뮬레이션(LLM) 방식.
- 선택형 문항은 Prolific 2026 방식: 보기별 선택 확률분포를 추출 후 샘플링.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import synth
from synth import (
    DEFAULT_PERSONA,
    DISCLAIMER,
    LIKERT_OPTIONS,
    METHODOLOGY_NOTE,
    QUESTION_TYPES,
    learn_distributions,
    new_question_id,
    question_summary,
)
from providers import PROVIDERS, auth_status

BASE_DIR = Path(__file__).parent
SAMPLE_Q_PATH = BASE_DIR / "samples" / "sample_questionnaire.json"
SAMPLE_XLSX_PATH = BASE_DIR / "samples" / "sample_real_data.xlsx"

STEPS = ["설문 설계", "페르소나 설계", "분포 학습", "가상 응답 생성", "결과"]

st.set_page_config(
    page_title="가상응답자 스튜디오",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1080px; padding-top: 2rem; }
    div[data-testid="stExpander"] { background: #ffffff; border-radius: 8px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 세션 상태 초기화
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "step": 1,
        "questions": [],
        "persona_dims": {k: dict(v) for k, v in DEFAULT_PERSONA.items()},
        "transcripts": [],
        "use_precision": False,
        "learned": None,
        "learned_file": None,
        "results_df": None,
        "gen_meta": None,
        "provider": "gemini",
        "model_override": "",
        "gen_cache": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def goto(step: int):
    st.session_state.step = step
    st.rerun()


def current_model() -> str:
    override = st.session_state.model_override.strip()
    if override:
        return override
    return PROVIDERS[st.session_state.provider]["default_model"]


# ----------------------------------------------------------------------------
# 사이드바
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("가상응답자 스튜디오")
    st.caption("Synthetic Respondent Studio")
    st.divider()

    step_labels = [f"{i + 1}. {name}" for i, name in enumerate(STEPS)]
    selected = st.radio(
        "단계",
        options=list(range(1, 6)),
        format_func=lambda i: step_labels[i - 1],
        index=st.session_state.step - 1,
        label_visibility="collapsed",
        key="sidebar_step",
    )
    if selected != st.session_state.step:
        st.session_state.step = selected
        st.rerun()

    st.divider()
    st.subheader("AI 설정")
    ok, msg = auth_status(st.session_state.provider)
    st.write(f"선택: **{PROVIDERS[st.session_state.provider]['label']}**")
    st.caption(f"모델: {current_model()}")
    if ok:
        st.success(msg)
    else:
        st.error(msg)

    n_q = len(st.session_state.questions)
    n_t = len(st.session_state.transcripts)
    st.divider()
    st.caption(f"문항 {n_q}개 · 페르소나 차원 {len(st.session_state.persona_dims)}개")
    if st.session_state.learned:
        st.caption(f"분포 학습됨 ({st.session_state.learned_file})")
    if n_t:
        st.caption(f"인터뷰 트랜스크립트 {n_t}건")


step = st.session_state.step


# ----------------------------------------------------------------------------
# Step 1: 설문 설계
# ----------------------------------------------------------------------------
def step1():
    st.header("1. 설문 설계")
    st.caption("가상 응답을 생성할 설문 문항을 입력하세요.")

    questions = st.session_state.questions

    if not questions:
        with st.container(border=True):
            st.write("아직 문항이 없습니다. 샘플 설문으로 시작하거나 직접 추가하세요.")
            if st.button("샘플 설문 불러오기", type="primary"):
                import json

                st.session_state.questions = json.loads(SAMPLE_Q_PATH.read_text(encoding="utf-8"))
                st.rerun()

    for idx, q in enumerate(questions):
        with st.expander(question_summary(q), expanded=False):
            new_text = st.text_input("문항 내용", value=q["text"], key=f"qtext_{q['id']}")
            if new_text != q["text"]:
                q["text"] = new_text
            if q["type"] in ("single", "multi"):
                opts_raw = st.text_area(
                    "보기 (한 줄에 하나씩)",
                    value="\n".join(q.get("options", [])),
                    key=f"qopts_{q['id']}",
                )
                q["options"] = [o.strip() for o in opts_raw.split("\n") if o.strip()]
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("위로", key=f"qup_{q['id']}", disabled=idx == 0):
                    questions[idx - 1], questions[idx] = questions[idx], questions[idx - 1]
                    st.rerun()
            with c2:
                if st.button("아래로", key=f"qdown_{q['id']}", disabled=idx == len(questions) - 1):
                    questions[idx + 1], questions[idx] = questions[idx], questions[idx + 1]
                    st.rerun()
            with c3:
                if st.button("삭제", key=f"qdel_{q['id']}"):
                    questions.pop(idx)
                    st.rerun()

    st.subheader("문항 추가")
    with st.container(border=True):
        with st.form("add_question"):
            qtext = st.text_input("문항 내용")
            qtype = st.selectbox(
                "문항 유형",
                options=list(QUESTION_TYPES.keys()),
                format_func=lambda k: QUESTION_TYPES[k],
            )
            qopts = ""
            if qtype in ("single", "multi"):
                qopts = st.text_area("보기 (한 줄에 하나씩 입력)")
            submitted = st.form_submit_button("추가", type="primary")
        if submitted:
            if not qtext.strip():
                st.error("문항 내용을 입력하세요.")
            elif qtype in ("single", "multi"):
                opts = [o.strip() for o in qopts.split("\n") if o.strip()]
                if len(opts) < 2:
                    st.error("보기를 2개 이상 입력하세요.")
                else:
                    st.session_state.questions.append(
                        {"id": new_question_id(questions), "text": qtext.strip(),
                         "type": qtype, "options": opts}
                    )
                    st.rerun()
            else:
                st.session_state.questions.append(
                    {"id": new_question_id(questions), "text": qtext.strip(),
                     "type": qtype, "options": []}
                )
                st.rerun()

    st.divider()
    col1, col2 = st.columns([1, 5])
    with col1:
        st.button(
            "다음: 페르소나 설계 →",
            type="primary",
            disabled=not questions,
            on_click=lambda: goto(2),
        )
    if not questions:
        with col2:
            st.caption("문항을 1개 이상 추가해야 다음 단계로 이동할 수 있습니다.")


# ----------------------------------------------------------------------------
# Step 2: 페르소나 설계
# ----------------------------------------------------------------------------
def step2():
    st.header("2. 페르소나 설계")
    st.caption("가상 응답자의 인구통계 분포를 설정합니다. 항목별 비율의 합은 100%여야 합니다.")

    dims = st.session_state.persona_dims
    valid = True

    for dim in list(dims.keys()):
        with st.container(border=True):
            h1, h2 = st.columns([5, 1])
            with h1:
                st.subheader(dim)
            with h2:
                if st.button("차원 삭제", key=f"dimdel_{dim}"):
                    del dims[dim]
                    st.rerun()
            cats = dims[dim]
            cols = st.columns(max(1, len(cats)))
            for i, cat in enumerate(list(cats.keys())):
                with cols[i % len(cols)]:
                    v = st.number_input(
                        cat, min_value=0.0, max_value=100.0, value=float(cats[cat]),
                        step=1.0, key=f"pct_{dim}_{cat}",
                    )
                    cats[cat] = v
            total = sum(cats.values())
            if abs(total - 100.0) > 0.01:
                st.error(f"합계가 100%가 아닙니다 (현재 {total:.1f}%).")
                valid = False
            else:
                st.success(f"합계 100% 확인")
            c1, c2 = st.columns(2)
            with c1:
                new_cat = st.text_input("구분 추가", key=f"newcat_{dim}", placeholder="예: 70대")
            with c2:
                st.write("")
                if st.button("구분 추가 실행", key=f"addcat_{dim}") and new_cat.strip():
                    cats[new_cat.strip()] = 0.0
                    st.rerun()
            for cat in list(cats.keys()):
                if st.button(f"'{cat}' 삭제", key=f"catdel_{dim}_{cat}"):
                    del cats[cat]
                    st.rerun()

    with st.expander("새 인구통계 차원 추가"):
        new_dim = st.text_input("차원 이름", placeholder="예: 소득대")
        if st.button("차원 추가") and new_dim.strip():
            dims[new_dim.strip()] = {"구분1": 50.0, "구분2": 50.0}
            st.rerun()

    st.subheader("고정밀 모드 (인터뷰 기반)")
    with st.container(border=True):
        st.caption(
            "스탠포드 Generative Agents 연구에서 정규화 정확도 83%로 가장 높았던 방식입니다. "
            "1인당 약 2시간의 인터뷰 트랜스크립트가 필요합니다. "
            "트랜스크립트를 업로드하면 생성 단계에서 고정밀 모드를 켤 수 있습니다."
        )
        files = st.file_uploader(
            "인터뷰 트랜스크립트 업로드 (.txt, .md)",
            type=["txt", "md"],
            accept_multiple_files=True,
            key="transcript_uploader",
        )
        if files:
            loaded = []
            for f in files:
                try:
                    text = f.read().decode("utf-8", errors="replace")
                except Exception:
                    text = ""
                loaded.append({"name": f.name, "text": text})
            st.session_state.transcripts = loaded
            st.success(f"{len(loaded)}건 로드됨")
        elif st.session_state.transcripts:
            st.info(f"{len(st.session_state.transcripts)}건 로드됨")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.button("← 이전: 설문 설계", on_click=lambda: goto(1))
    with c2:
        st.button("다음: 분포 학습 →", type="primary", disabled=not valid, on_click=lambda: goto(3))


# ----------------------------------------------------------------------------
# Step 3: 분포 학습
# ----------------------------------------------------------------------------
def step3():
    st.header("3. 분포 학습 (선택)")
    st.caption("실제 응답 엑셀을 올리면 문항별 응답 분포를 학습합니다. 건너뛰면 페르소나 기반 생성만 합니다.")

    with st.expander("방법론 안내", expanded=True):
        st.markdown(METHODOLOGY_NOTE.replace("\n", "  \n"))

    uploaded = st.file_uploader("실제 응답 엑셀 업로드 (.xlsx)", type=["xlsx"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("샘플 데이터로 체험"):
            df = pd.read_excel(SAMPLE_XLSX_PATH)
            st.session_state.learned = learn_distributions(df, st.session_state.questions)
            st.session_state.learned_file = "sample_real_data.xlsx (샘플)"
            st.rerun()
    with c2:
        if st.button("분포 학습 초기화"):
            st.session_state.learned = None
            st.session_state.learned_file = None
            st.rerun()

    if uploaded is not None:
        try:
            df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"엑셀 읽기 실패: {e}")
            df = None
        if df is not None:
            with st.spinner("분포 학습 중..."):
                learned = learn_distributions(df, st.session_state.questions)
            st.session_state.learned = learned
            st.session_state.learned_file = uploaded.name
            st.success(f"{len(learned)}개 문항 매칭됨")

    learned = st.session_state.learned
    if learned:
        st.subheader(f"학습된 분포 ({st.session_state.learned_file})")
        for q in st.session_state.questions:
            info = learned.get(q["id"])
            with st.container(border=True):
                if not info:
                    st.write(f"{question_summary(q)} — 매칭된 컬럼 없음")
                    continue
                st.write(f"**{question_summary(q)}**  ·  n={info['n']}")
                if info["dist"]:
                    ddf = pd.DataFrame(
                        {"보기": list(info["dist"].keys()), "비율": list(info["dist"].values())}
                    )
                    fig = px.bar(ddf, x="보기", y="비율", text_auto=".1%")
                    fig.update_layout(height=280, margin=dict(t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("주관식 문항 — 분포 학습 대상 아님")
    else:
        st.info("학습된 분포가 없습니다. 생성 단계에서는 페르소나 + LLM 분포 추정 방식으로 생성합니다.")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.button("← 이전: 페르소나 설계", on_click=lambda: goto(2))
    with c2:
        st.button("다음: 가상 응답 생성 →", type="primary", on_click=lambda: goto(4))


# ----------------------------------------------------------------------------
# Step 4: 가상 응답 생성
# ----------------------------------------------------------------------------
def step4():
    st.header("4. 가상 응답 생성")
    st.caption("AI로 가상 응답자를 생성합니다.")

    with st.container(border=True):
        st.subheader("AI 선택")
        provider = st.radio(
            "사용할 AI",
            options=list(PROVIDERS.keys()),
            format_func=lambda k: f"{PROVIDERS[k]['label']} ({PROVIDERS[k]['default_model']})",
            horizontal=True,
            key="provider_radio",
        )
        st.session_state.provider = provider
        ok, msg = auth_status(provider)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
        model = st.text_input(
            "모델명 (비워두면 기본값)",
            value=st.session_state.model_override,
            placeholder=PROVIDERS[provider]["default_model"],
            key="model_input",
        )
        st.session_state.model_override = model
        if provider == "gemini":
            st.caption(
                "Gemini 기본값은 Pro 계열입니다 (1,000개 설문 벤치마크에서 표준 LLM 중 최고 67%). "
                "무료 등급에서 할당량 오류(429)가 나면 gemini-3.6-flash를 입력하세요."
            )

    with st.container(border=True):
        st.subheader("생성 방식")
        has_learned = bool(st.session_state.learned)
        st.write(
            "- 학습된 분포가 있는 선택형 문항: 경험분포에서 직접 샘플링 (API 호출 없음)"
            if has_learned
            else "- 선택형 문항: Prolific 2026 방식 — LLM이 페르소나별 보기별 선택 확률분포를 추정하면 샘플링 (분포 오차 35~48% 감소 보고)"
        )
        st.write("- 주관식 문항: LLM 배치 생성")
        n_trans = len(st.session_state.transcripts)
        use_precision = st.checkbox(
            "고정밀 모드 (인터뷰 기반)",
            value=st.session_state.use_precision and n_trans > 0,
            disabled=n_trans == 0,
            help="스탠포드 Generative Agents 연구: 정규화 정확도 83%. 트랜스크립트 업로드 시 활성화.",
        )
        st.session_state.use_precision = use_precision and n_trans > 0
        if n_trans == 0:
            st.caption("인터뷰 트랜스크립트를 업로드하면 활성화됩니다 (2단계에서 업로드).")

    n = st.number_input("생성할 응답자 수", min_value=1, max_value=200, value=30, step=1)

    can_run = ok and len(st.session_state.questions) > 0
    if st.button("생성 시작", type="primary", disabled=not can_run):
        if not ok:
            st.error("AI 인증이 필요합니다.")
        else:
            progress = st.progress(0.0, text="생성 중...")
            try:
                df = synth.generate_responses(
                    questions=st.session_state.questions,
                    persona_dims=st.session_state.persona_dims,
                    n=int(n),
                    provider_id=provider,
                    model=current_model(),
                    learned=st.session_state.learned,
                    progress_cb=lambda f: progress.progress(f, text=f"생성 중... {int(f * 100)}%"),
                    cache=st.session_state.gen_cache,
                    transcripts=st.session_state.transcripts if st.session_state.use_precision else None,
                )
                st.session_state.results_df = df
                st.session_state.gen_meta = {
                    "생성일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "AI": PROVIDERS[provider]["label"],
                    "모델": current_model(),
                    "응답자 수": int(n),
                    "분포 학습": "사용" if has_learned else "미사용",
                    "고정밀 모드": "사용" if st.session_state.use_precision else "미사용",
                }
                progress.progress(1.0, text="완료")
                st.success(f"{len(df)}건 생성 완료")
            except Exception as e:
                st.error(f"생성 실패: {e}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.button("← 이전: 분포 학습", on_click=lambda: goto(3))
    with c2:
        st.button(
            "다음: 결과 확인 →",
            type="primary",
            disabled=st.session_state.results_df is None,
            on_click=lambda: goto(5),
        )


# ----------------------------------------------------------------------------
# Step 5: 결과
# ----------------------------------------------------------------------------
def step5():
    st.header("5. 결과")

    # 정직한 한계 고지 — 결과 탭 상단 고정 표시
    st.warning(DISCLAIMER)

    df = st.session_state.results_df
    if df is None:
        st.info("아직 생성된 데이터가 없습니다. 4단계에서 가상 응답을 생성하세요.")
        if st.button("← 생성 단계로 이동", on_click=lambda: goto(4)):
            pass
        return

    meta = st.session_state.gen_meta or {}
    with st.container(border=True):
        cols = st.columns(len(meta) or 1)
        for i, (k, v) in enumerate(meta.items()):
            cols[i].metric(k, v)

    st.subheader("데이터 미리보기")
    st.dataframe(df, use_container_width=True, height=400)

    # 생성 분포 vs 학습 분포 비교
    learned = st.session_state.learned or {}
    comp_qs = [
        q for q in st.session_state.questions
        if q["type"] in ("single", "likert") and q["id"] in learned and learned[q["id"]].get("dist")
    ]
    if comp_qs:
        st.subheader("분포 비교 (생성 vs 학습)")
        for q in comp_qs:
            with st.container(border=True):
                st.write(f"**{question_summary(q)}**")
                gen_counts = df[q["id"]].astype(str).value_counts(normalize=True)
                cats = list(learned[q["id"]]["dist"].keys())
                cdf = pd.DataFrame({
                    "보기": cats,
                    "학습 분포": [learned[q["id"]]["dist"][c] for c in cats],
                    "생성 분포": [float(gen_counts.get(c, 0)) for c in cats],
                })
                mdf = cdf.melt(id_vars="보기", var_name="구분", value_name="비율")
                fig = px.bar(mdf, x="보기", y="비율", color="구분", barmode="group", text_auto=".1%")
                fig.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("다운로드")
    c1, c2 = st.columns(2)
    with c1:
        xlsx = synth.to_excel_bytes(df, meta=meta)
        st.download_button(
            "엑셀 다운로드 (.xlsx)",
            data=xlsx,
            file_name="synthetic_responses.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        st.caption("방법론_안내 시트(한계 고지 포함)가 함께 저장됩니다.")
    with c2:
        st.download_button(
            "CSV 다운로드 (.csv)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="synthetic_responses.csv",
            mime="text/csv",
        )

    st.divider()
    if st.button("← 이전: 가상 응답 생성", on_click=lambda: goto(4)):
        pass


# ----------------------------------------------------------------------------
# 라우팅
# ----------------------------------------------------------------------------
{"step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5}[
    f"step{step}"
]()
