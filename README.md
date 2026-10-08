# 가상응답자 스튜디오 (Synthetic Respondent Studio)

설문 문항 설계 → 페르소나 설계 → 분포 학습 → 가상 응답 생성 → 결과 확인·다운로드까지
이어지는 5단계 마법사형 Streamlit 웹앱입니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

### 로컬 AI 인증

이 앱은 API 키를 코드에 넣지 않습니다. 로컬에서는 아래 CLI의 기존 인증 방식을 그대로 사용합니다.

- Gemini: `~/workspace/skills/gemini/bin/gemini`
- OpenAI: `~/workspace/skills/openai/bin/chatgpt`

두 CLI가 존재하면 별도 설정 없이 바로 생성 기능을 쓸 수 있습니다.

## Streamlit Cloud 배포

1. 이 폴더를 GitHub 리포지토리에 푸시합니다.
2. [share.streamlit.io](https://share.streamlit.io) 에서 **New app** → 리포지토리·브랜치·`app.py` 선택 후 **Deploy**.
3. 배포된 앱의 **Settings → Secrets** 에 아래를 등록합니다.

```toml
GEMINI_API_KEY = "발급받은_Gemini_API_키"
OPENAI_API_KEY = "발급받은_OpenAI_API_키"
```

4. 앱이 재시작되면 생성 단계에서 AI를 선택해 사용할 수 있습니다.

로컬에서 secrets 방식을 시험하려면:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# .streamlit/secrets.toml 에 실제 키 입력 (이 파일은 .gitignore 처리됨)
```

## 방법론과 한계 (정직한 고지)

- **파인튜닝이 아닙니다.** GPU 가중치 학습을 하지 않습니다.
  - **분포 학습**: 업로드된 실제 응답 엑셀에서 문항별 경험분포를 추출합니다.
  - **페르소나 시뮬레이션**: LLM에 페르소나 프로필을 주고 가상 응답을 생성하게 합니다.
- **선택형 문항 생성 (Prolific 2026 연구 반영)**: 페르소나마다 강제 단일 응답을 뽑지 않고,
  LLM으로부터 보기별 선택 확률분포를 추출한 뒤 샘플링합니다. 연구 보고에 따르면
  분포 오차를 35~48% 줄인다고 합니다.
- **기본 모델**: Gemini Pro 계열 (`gemini-pro-latest`)을 기본값으로 합니다.
  1,000개 설문 공개 벤치마크에서 표준 LLM 중 최고(67%, GPT-5 62% 대비)라는
  리서치를 반영한 선택입니다. 무료 등급에서 할당량 오류(429)가 나면
  `gemini-3.6-flash` 같은 경량 모델로 바꾸세요.
- **고정밀 모드 (인터뷰 기반, 선택)**: 스탠포드 Generative Agents 연구에서
  정규화 정확도 83%로 가장 높았던 방식입니다. 인터뷰 트랜스크립트를 업로드하면
  생성 단계에서 켤 수 있습니다. 1인당 약 2시간의 인터뷰가 필요합니다.
- **한계**: 합성응답자는 탐색적·방향성 리서치용이며 인간 실측을 대체할 수 없습니다
  (Pew: 평균 12%p 오차). 이 문구는 결과 화면 상단과 엑셀 다운로드의
  `방법론_안내` 시트에 고정 표시됩니다.

## 파일 구성

```
app.py                  # Streamlit 마법사 UI (렌더링만 — 로직은 core/ 호출)
config.py               # 중앙 설정 (상수·모델명·기본 분포·고지문)
core/
  __init__.py           # 공개 API 진입점
  models.py             # Question·LearnedDist 등 데이터 모델
  question_types.py     # 문항 유형 레지스트리 (플러그인 ①)
  providers.py          # AI provider 레지스트리 (플러그인 ②)
  strategies.py         # 생성 전략 레지스트리 (플러그인 ③)
  persona.py            # 페르소나 샘플링
  services.py           # Questionnaire·Learning·Generation·Export 서비스
  store.py              # Store 추상화 (SessionStore/MemoryStore)
tests/
  test_registries.py    # 플러그인 등록·오류 메시지 테스트
  test_core.py          # 페르소나·학습·생성·엑셀 테스트 (LLM 모킹)
requirements.txt        # streamlit, pandas, openpyxl, plotly
.streamlit/config.toml  # 테마 설정
.streamlit/secrets.toml.example  # 로컬/클라우드 secrets 템플릿 (실제 키 없음)
.gitignore
samples/
  sample_questionnaire.json  # 데모 설문 (온라인 쇼핑 만족도, 6문항)
  sample_real_data.xlsx      # 데모 실제응답 200건
  make_samples.py            # 샘플 재생성 스크립트
```

## 아키텍처: 플러그인 포인트 3종

**① 새 AI provider 추가** — `core/providers.py`
1. `AIProvider`를 상속한 클래스 작성 (`_call_rest`만 구현)
2. `_PROVIDER_CLASSES`에 `{"kind": 새클래스}` 한 줄 추가
3. `config.py`의 `PROVIDER_SPECS`에 스펙(라벨·모델·CLI·시크릿명) 한 줄 추가
4. 호출부의 `if provider == ...` 분기는 없음 — 레지스트리가 디스패치
5. `tests/test_registries.py`에 등록 테스트 1개 추가 권장

**② 새 문항 유형 추가** — `core/question_types.py`
1. `QuestionType`을 상속한 클래스 작성
   (`validate`·`prompt_fragment`·`parse_response`·`learn_distribution` 구현)
2. 클래스 위에 `@register_question_type` 데코레이터 1줄
3. 생성 전략·분포 학습·UI가 자동으로 새 유형 인식 (추가 수정 불필요)
4. 선택형이면 `is_choice = True` — 분포추출/학습 전략이 자동 매칭
5. `tests/test_registries.py`에 등록 테스트 1개 추가 권장

**③ 새 생성 전략 추가** — `core/strategies.py`
1. `GenerationStrategy`를 상속한 클래스 작성 (`supports`·`generate` 구현)
2. 하단에 `register_strategy(새전략())` 한 줄 추가
3. (선택) `config.py`의 `STRATEGY_ORDER`에 우선순위 반영
4. `GenerationService`가 문항마다 우선순위대로 자동 매칭
5. `generate_fn` 주입으로 LLM 없이 테스트 가능 (`GenerationContext`)

## 사용 흐름

1. **설문 설계**: 문항 입력 (단일선택/다중선택/5점 리커트/주관식). 샘플 설문 불러오기 가능.
2. **페르소나 설계**: 성별·연령대·지역·직업 등 항목별 비율 입력 (합계 100% 검증).
   인터뷰 트랜스크립트 업로드 시 고정밀 모드 사용 가능.
3. **분포 학습 (선택)**: 실제 응답 엑셀 업로드 → 문항별 분포 시각화. 샘플 데이터로 체험 가능.
4. **가상 응답 생성**: AI 선택 (Gemini/OpenAI), 응답자 수 입력, 진행률 표시.
5. **결과**: 한계 고지 → 데이터 미리보기 → 생성/학습 분포 비교 → 엑셀/CSV 다운로드.
