"""가상응답자 스튜디오 (Synthetic Respondent Studio)

설문 설계 → 페르소나 설계 → 분포 학습 → 가상 응답 생성 → 결과
5단계 마법사형 Streamlit 앱.

아키텍처: 이 파일은 렌더링만 담당한다. 비즈니스 로직은 core/ 패키지의
서비스(QuestionnaireService·LearningService·GenerationService·ExportService)가
처리하고, 상태 접근은 Store 인터페이스(SessionStore)를 통한다.
"""

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (APP_SUBTITLE, APP_TITLE, DEFAULT_PERSONA, DISCLAIMER,
                    METHODOLOGY_NOTE, SAMPLE_XLSX_PATH)
from core import (QUESTION_TYPE_REGISTRY, PROVIDER_REGISTRY, ExportService, GenerationService,
                  LearningService, PdfQuestionnaireService, QuestionnaireService, SessionStore,
                  auth_status, get_provider, get_question_type)

STEPS = ["설문 설계", "페르소나 설계", "분포 학습", "가상 응답 생성", "결과"]

# ----------------------------------------------------------------------------
# 서비스 · 저장소 (UI는 이 진입점만 쓴다)
# ----------------------------------------------------------------------------
store = SessionStore()
q_service = QuestionnaireService()
learn_service = LearningService()
gen_service = GenerationService()
export_service = ExportService()

st.set_page_config(
    page_title=APP_TITLE,
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
# 상태 초기화 (Store 경유 — 향후 DB 저장소로 교체 가능)
# ----------------------------------------------------------------------------
def init_state():
    store.set_default("step", 1)
    store.set_default("questions", [])
    store.set_default("persona_dims",
                      {k: dict(v) for k, v in DEFAULT_PERSONA.items()})
    store.set_default("transcripts", [])
    store.set_default("use_precision", False)
    store.set_default("learned", None)
    store.set_default("learned_file", None)
    store.set_default("results_df", None)
    store.set_default("gen_meta", None)
    store.set_default("provider", "gemini")
    store.set_default("model_override", "")
    store.set_default("gen_cache", {})
    store.set_default("show_theory", False)


init_state()


def goto(step: int):
    store.set("step", step)
    # 사이드바 radio 위젯 상태도 함께 동기화 (위젯 state가 우선되어 되돌아가는 버그 방지)
    st.session_state["sidebar_step"] = step
    st.rerun()


def current_model() -> str:
    override = (store.get("model_override") or "").strip()
    if override:
        return override
    return get_provider(store.get("provider")).default_model


# ----------------------------------------------------------------------------
# 사이드바
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.divider()

    step_labels = [f"{i + 1}. {name}" for i, name in enumerate(STEPS)]
    selected = st.radio(
        "단계",
        options=list(range(1, 6)),
        format_func=lambda i: step_labels[i - 1],
        index=store.get("step") - 1,
        label_visibility="collapsed",
        key="sidebar_step",
    )
    if selected != store.get("step"):
        store.set("step", selected)
        st.rerun()

    st.divider()
    st.subheader("AI 설정")
    ok, msg = auth_status(store.get("provider"))
    st.write(f"선택: **{get_provider(store.get('provider')).label}**")
    st.caption(f"모델: {current_model()}")
    if ok:
        st.success(msg)
    else:
        st.error(msg)

    n_q = len(store.get("questions"))
    n_t = len(store.get("transcripts"))
    st.divider()
    st.caption(f"문항 {n_q}개 · 페르소나 차원 {len(store.get('persona_dims'))}개")
    if store.get("learned"):
        st.caption(f"분포 학습됨 ({store.get('learned_file')})")
    if n_t:
        st.caption(f"인터뷰 트랜스크립트 {n_t}건")

    st.divider()
    if st.button("📖 이론적 배경", use_container_width=True):
        store.set("show_theory", True)
        st.rerun()


step = store.get("step")


# ----------------------------------------------------------------------------
# 이론적 배경
# ----------------------------------------------------------------------------
def theory_page():
    if st.button("← 돌아가기"):
        store.set("show_theory", False)
        st.rerun()

    st.header("📖 이론적 배경")
    st.caption("합성응답자(Synthetic Respondents) 방법론의 이론적 기초")

    st.subheader("1. 합성응답자란?")
    st.write(
        "합성응답자(Synthetic Respondents)는 대규모 언어모델(LLM)을 이용해 "
        "특정 인구통계학적·심리적 프로필을 가진 가상 인물을 시뮬레이션하고, "
        "이들이 설문 문항에 어떻게 응답할지를 생성하는 방법론입니다. "
        "실제 사람을 대상으로 조사하기 전에 설문지의 품질을 검증하거나, "
        "연구 방향성을 탐색하는 **파일럿 테스트** 용도로 활용됩니다."
    )

    st.subheader("2. 이론적 기반")
    st.markdown(
        """
**① LLM as Human Simulators**

LLM은 방대한 인간 생성 텍스트로 학습되면서 인간의 응답 패턴, 편향, 맥락적 뉘앙스를
내재화합니다. 적절한 페르소나 프롬프팅을 통해 특정 집단의 응답 성향을 모사할 수 있다는 것이
최근 연구들의 공통된 발견입니다.

**② 분포 기반 샘플링 (Prolific 2026)**

본 도구는 Prolific 2026 연구 방식을 따릅니다. 페르소나에게 강제 단일 응답을 요구하는 대신,
보기별 선택 **확률분포**를 LLM으로부터 추출한 뒤 샘플링합니다.
이 방식은 단일 응답 방식 대비 분포 오차를 35~48% 감소시키는 것으로 보고되었습니다.

**③ 고정밀 모드 (Stanford Generative Agents)**

인터뷰 트랜스크립트를 활용하는 고정밀 모드는 스탠포드 Generative Agents 연구에 기반합니다.
2시간 분량의 심층 인터뷰를 바탕으로 생성된 에이전트는
정규화 정확도 83%로 가장 높은 fidelity를 보였습니다.
"""
    )

    st.subheader("3. 본 도구의 방법론")
    st.markdown(
        """
본 도구는 **GPU 파인튜닝(모델 가중치 학습)을 하지 않습니다.** 대신 3단계 파이프라인으로 동작합니다.

**Step A. 분포 학습**
업로드된 실제 응답 데이터(Excel)에서 문항별 경험분포를 추출합니다.
이는 모집단의 응답 경향성에 대한 empirical prior 역할을 합니다.

**Step B. 페르소나 설계**
연령, 성별, 직업 등 인구통계 차원별 분포를 설정합니다.
각 가상 응답자는 이 분포에서 샘플링된 프로필을 부여받습니다.

**Step C. 확률분포 샘플링**
각 페르소나에 대해 LLM으로부터 보기별 선택 확률을 추출한 뒤,
이 분포에서 실제 응답을 샘플링합니다.
주관식 문항은 페르소나 프로필을 조건으로 직접 생성합니다.
"""
    )

    st.subheader("4. 한계점")
    st.warning(
        "합성응답자는 탐색적·방향성 리서치용이며 인간 실측을 대체할 수 없습니다. "
        "(Pew Research: 평균 12%p 오차)"
    )
    st.markdown(
        """
- **생태학적 타당도 한계**: LLM의 학습 데이터에 없는 새로운 현상이나 급변하는 여론은 정확히 모사하기 어렵습니다.
- **편향 증폭 위험**: 학습 데이터의 편향이 합성 응답에 그대로 반영될 수 있습니다.
- **과적합 주의**: 소수 실제 데이터에 과도하게 맞추면 일반화 성능이 떨어집니다.
- **용도 제한**: 본조사 대체가 아닌, 본조사 **전** 파일럿 검증용으로만 사용해야 합니다.
"""
    )

    st.subheader("5. 참고 문헌")
    st.markdown(
        """
- Prolific (2026). 분포 기반 합성응답 생성 방법론.
- Park et al. Stanford Generative Agents 연구 (정규화 정확도 83%).
- Pew Research Center. 합성응답자와 인간 실측 비교 (평균 12%p 오차).
- Gemini Pro: 1,000개 설문 공개 벤치마크에서 표준 LLM 중 최고 성능 (67%).
"""
    )


step = store.get("step")


# ----------------------------------------------------------------------------
# Step 1: 설문 설계
# ----------------------------------------------------------------------------
def step1():
    st.header("1. 설문 설계")
    st.caption("가상 응답을 생성할 설문 문항을 입력하세요.")

    questions = store.get("questions")

    if not questions:
        with st.container(border=True):
            st.write("아직 문항이 없습니다. 샘플 설문으로 시작하거나 직접 추가하세요.")
            if st.button("샘플 설문 불러오기", type="primary"):
                store.set("questions", q_service.load_sample_questions())
                st.rerun()

    # PDF 설문지 자동 인식
    with st.expander("📄 PDF 설문지에서 자동 인식", expanded=False):
        st.caption("PDF 설문지를 업로드하면 AI가 문항·보기·분기 로직을 자동 파싱합니다.")
        pdf_file = st.file_uploader("PDF 파일 선택", type=["pdf"], key="pdf_upload")
        if pdf_file is not None:
            if st.button("문항 자동 인식 시작", type="primary", key="pdf_parse"):
                with st.spinner("PDF를 분석하고 있습니다..."):
                    try:
                        pdf_text = PdfQuestionnaireService.extract_text(pdf_file.read())
                        if not pdf_text.strip():
                            st.error("PDF에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF는 지원하지 않습니다.")
                        else:
                            parsed = PdfQuestionnaireService.parse_questions(
                                pdf_text, store.get("provider"))
                            if not parsed:
                                st.error("문항 파싱에 실패했습니다. PDF 내용을 확인해주세요.")
                            else:
                                # 기존 문항에 추가 (ID 부여)
                                for pq in parsed:
                                    pq["id"] = q_service.new_question_id(questions)
                                    questions.append(pq)
                                store.set("questions", questions)
                                n_branch = sum(1 for q in parsed if q.get("branching"))
                                st.success(f"{len(parsed)}개 문항을 인식했습니다. (분기 로직 {n_branch}건 포함)")
                                st.rerun()
                    except Exception as e:
                        st.error(f"처리 중 오류: {e}")

    for idx, q in enumerate(questions):
        with st.expander(q_service.question_summary(q), expanded=False):
            new_text = st.text_input("문항 내용", value=q["text"], key=f"qtext_{q['id']}")
            if new_text != q["text"]:
                q["text"] = new_text
                store.set("questions", questions)
            qtype = get_question_type(q["type"])
            if qtype.needs_options:
                opts_raw = st.text_area(
                    "보기 (한 줄에 하나씩)",
                    value="\n".join(q.get("options", [])),
                    key=f"qopts_{q['id']}",
                )
                q["options"] = [o.strip() for o in opts_raw.split("\n") if o.strip()]
                store.set("questions", questions)
            # 분기 로직 표시/편집
            branching = st.text_input(
                "분기 로직 (예: 문3에서 ① 응답 시 문5로 이동)",
                value=q.get("branching", ""),
                key=f"qbranch_{q['id']}",
            )
            if branching != q.get("branching", ""):
                q["branching"] = branching
                store.set("questions", questions)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("위로", key=f"qup_{q['id']}", disabled=idx == 0):
                    questions[idx - 1], questions[idx] = questions[idx], questions[idx - 1]
                    store.set("questions", questions)
                    st.rerun()
            with c2:
                if st.button("아래로", key=f"qdown_{q['id']}", disabled=idx == len(questions) - 1):
                    questions[idx + 1], questions[idx] = questions[idx], questions[idx + 1]
                    store.set("questions", questions)
                    st.rerun()
            with c3:
                if st.button("삭제", key=f"qdel_{q['id']}"):
                    questions.pop(idx)
                    store.set("questions", questions)
                    st.rerun()

    st.subheader("문항 추가")
    # 문항 유형 선택을 폼 바깥에 두어 변경 시 즉시 리런되도록 함
    # (폼 안에서는 selectbox 변경이 리런을 유발하지 않아 보기 입력란이 갱신되지 않는 버그)
    qtype_name = st.selectbox(
        "문항 유형",
        options=list(QUESTION_TYPE_REGISTRY.keys()),
        format_func=lambda k: QUESTION_TYPE_REGISTRY[k].label,
        key="add_qtype",
    )
    with st.container(border=True):
        with st.form("add_question"):
            qtext = st.text_input("문항 내용")
            qopts = ""
            if get_question_type(qtype_name).needs_options:
                qopts = st.text_area("보기 (한 줄에 하나씩 입력)")
            qbranch = st.text_input("분기 로직 (선택사항)")
            submitted = st.form_submit_button("추가", type="primary")
        if submitted:
            opts = [o.strip() for o in qopts.split("\n") if o.strip()]
            qdict = {
                "id": q_service.new_question_id(questions),
                "text": qtext.strip(),
                "type": qtype_name,
                "options": opts,
                "branching": qbranch.strip(),
            }
            errs = q_service.validate_question(qdict)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                questions.append(qdict)
                store.set("questions", questions)
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

    dims = store.get("persona_dims")
    valid = True

    for dim in list(dims.keys()):
        with st.container(border=True):
            h1, h2 = st.columns([5, 1])
            with h1:
                st.subheader(dim)
            with h2:
                if st.button("차원 삭제", key=f"dimdel_{dim}"):
                    del dims[dim]
                    store.set("persona_dims", dims)
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
            store.set("persona_dims", dims)
            total = sum(cats.values())
            if abs(total - 100.0) > 0.01:
                st.error(f"합계가 100%가 아닙니다 (현재 {total:.1f}%).")
                valid = False
            else:
                st.success("합계 100% 확인")
            c1, c2 = st.columns(2)
            with c1:
                new_cat = st.text_input("구분 추가", key=f"newcat_{dim}", placeholder="예: 70대")
            with c2:
                st.write("")
                if st.button("구분 추가 실행", key=f"addcat_{dim}") and new_cat.strip():
                    cats[new_cat.strip()] = 0.0
                    store.set("persona_dims", dims)
                    st.rerun()
            for cat in list(cats.keys()):
                if st.button(f"'{cat}' 삭제", key=f"catdel_{dim}_{cat}"):
                    del cats[cat]
                    store.set("persona_dims", dims)
                    st.rerun()

    with st.expander("새 인구통계 차원 추가"):
        new_dim = st.text_input("차원 이름", placeholder="예: 소득대")
        if st.button("차원 추가") and new_dim.strip():
            dims[new_dim.strip()] = {"구분1": 50.0, "구분2": 50.0}
            store.set("persona_dims", dims)
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
            store.set("transcripts", loaded)
            st.success(f"{len(loaded)}건 로드됨")
        elif store.get("transcripts"):
            st.info(f"{len(store.get('transcripts'))}건 로드됨")

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
            store.set("learned",
                      learn_service.learn_distributions(df, store.get("questions")))
            store.set("learned_file", "sample_real_data.xlsx (샘플)")
            st.rerun()
    with c2:
        if st.button("분포 학습 초기화"):
            store.set("learned", None)
            store.set("learned_file", None)
            st.rerun()

    if uploaded is not None:
        try:
            df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"엑셀 읽기 실패: {e}")
            df = None
        if df is not None:
            with st.spinner("분포 학습 중..."):
                learned = learn_service.learn_distributions(df, store.get("questions"))
            store.set("learned", learned)
            store.set("learned_file", uploaded.name)
            st.success(f"{len(learned)}개 문항 매칭됨")

    learned = store.get("learned")
    if learned:
        st.subheader(f"학습된 분포 ({store.get('learned_file')})")
        for q in store.get("questions"):
            info = learned.get(q["id"])
            with st.container(border=True):
                if not info:
                    st.write(f"{q_service.question_summary(q)} — 매칭된 컬럼 없음")
                    continue
                st.write(f"**{q_service.question_summary(q)}**  ·  n={info['n']}")
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
            options=list(PROVIDER_REGISTRY.keys()),
            format_func=lambda k: f"{get_provider(k).label} ({get_provider(k).default_model})",
            horizontal=True,
            key="provider",
        )
        ok, msg = auth_status(provider)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
        st.text_input(
            "모델명 (비워두면 기본값)",
            placeholder=get_provider(provider).default_model,
            key="model_override",
        )
        if provider == "gemini":
            st.caption(
                "Gemini 기본값은 Pro 계열입니다 (1,000개 설문 벤치마크에서 표준 LLM 중 최고 67%). "
                "무료 등급에서 할당량 오류(429)가 나면 gemini-3.6-flash를 입력하세요."
            )

    with st.container(border=True):
        st.subheader("생성 방식")
        has_learned = bool(store.get("learned"))
        st.write(
            "- 학습된 분포가 있는 선택형 문항: 경험분포에서 직접 샘플링 (API 호출 없음)"
            if has_learned
            else "- 선택형 문항: Prolific 2026 방식 — LLM이 페르소나별 보기별 선택 확률분포를 추정하면 샘플링 (분포 오차 35~48% 감소 보고)"
        )
        st.write("- 주관식 문항: LLM 배치 생성")
        n_trans = len(store.get("transcripts"))
        use_precision = st.checkbox(
            "고정밀 모드 (인터뷰 기반)",
            value=store.get("use_precision") and n_trans > 0,
            disabled=n_trans == 0,
            help="스탠포드 Generative Agents 연구: 정규화 정확도 83%. 트랜스크립트 업로드 시 활성화.",
        )
        store.set("use_precision", use_precision and n_trans > 0)
        if n_trans == 0:
            st.caption("인터뷰 트랜스크립트를 업로드하면 활성화됩니다 (2단계에서 업로드).")

    n = st.number_input("생성할 응답자 수", min_value=1, max_value=200, value=30, step=1)

    can_run = ok and len(store.get("questions")) > 0
    if st.button("생성 시작", type="primary", disabled=not can_run):
        if not ok:
            st.error("AI 인증이 필요합니다.")
        else:
            progress = st.progress(0.0, text="생성 중...")
            try:
                df = gen_service.generate_responses(
                    questions=store.get("questions"),
                    persona_dims=store.get("persona_dims"),
                    n=int(n),
                    provider_id=provider,
                    model=current_model(),
                    learned=store.get("learned"),
                    progress_cb=lambda f: progress.progress(f, text=f"생성 중... {int(f * 100)}%"),
                    cache=store.get("gen_cache"),
                    transcripts=store.get("transcripts") if store.get("use_precision") else None,
                    use_precision=store.get("use_precision"),
                )
                store.set("results_df", df)
                store.set("gen_meta", {
                    "생성일시": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "AI": get_provider(provider).label,
                    "모델": current_model(),
                    "응답자 수": int(n),
                    "분포 학습": "사용" if has_learned else "미사용",
                    "고정밀 모드": "사용" if store.get("use_precision") else "미사용",
                })
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
            disabled=store.get("results_df") is None,
            on_click=lambda: goto(5),
        )


# ----------------------------------------------------------------------------
# Step 5: 결과
# ----------------------------------------------------------------------------
def step5():
    st.header("5. 결과")

    # 정직한 한계 고지 — 결과 탭 상단 고정 표시
    st.warning(DISCLAIMER)

    df = store.get("results_df")
    if df is None:
        st.info("아직 생성된 데이터가 없습니다. 4단계에서 가상 응답을 생성하세요.")
        if st.button("← 생성 단계로 이동", on_click=lambda: goto(4)):
            pass
        return

    meta = store.get("gen_meta") or {}
    with st.container(border=True):
        cols = st.columns(len(meta) or 1)
        for i, (k, v) in enumerate(meta.items()):
            cols[i].metric(k, v)

    st.subheader("데이터 미리보기")
    st.dataframe(df, use_container_width=True, height=400)

    # 생성 분포 vs 학습 분포 비교
    learned = store.get("learned") or {}
    comp_qs = [
        q for q in store.get("questions")
        if q["type"] in ("single", "likert") and q["id"] in learned and learned[q["id"]].get("dist")
    ]
    if comp_qs:
        st.subheader("분포 비교 (생성 vs 학습)")
        for q in comp_qs:
            with st.container(border=True):
                st.write(f"**{q_service.question_summary(q)}**")
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
        xlsx = export_service.to_excel_bytes(df, meta=meta)
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
if store.get("show_theory"):
    theory_page()
else:
    {"step1": step1, "step2": step2, "step3": step3, "step4": step4, "step5": step5}[
        f"step{step}"
    ]()
