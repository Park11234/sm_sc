import streamlit as st
import streamlit.components.v1 as components

from LLM import (
    is_similar,
    generate_with_openai,
    generate_with_gemini,
    gather_context,
    extract_questions,
    parse_mc_questions,
    parse_eval,
    get_llm_backend,
    get_chat_llm,
    hist_pairs,
    hist_text,
    summarize_docs
)

st.set_page_config(page_title="확산", layout="wide")

st.header("4) 확산 (Diffusion)")

CATEGORY_NAME = "확산"

st.subheader("개요")
st.write(
    "고온 열처리를 통해 도펀트가 실리콘 벌크로 확산되어 농도 프로파일을 형성하는 공정입니다. "
    "프리디포지션(표면 농도/도즈 형성)과 드라이브인(접합 깊이 확보)으로 나뉘며, "
    "접합 깊이(xj), 시트저항(Rs), 표면 농도(Cs), 총 도즈(Q), 균일도, 활성화가 핵심 관리 항목입니다."
)

# 핵심 포인트
st.subheader("핵심 포인트")
st.markdown("""
- <span title="오염·수분·네이티브 산화막 상태를 통일하여 초기 계면 상태 최적화">전처리 세정</span> →
  <span title="도펀트 소스(고체/액체/기체)와 접촉하여 표면 농도 또는 도즈를 부여">프리디포지션</span> →
  <span title="고온에서 목표 접합 깊이까지 확산 진행, 프로파일 형상 제어">드라이브인</span> →
  <span title="산소/불활성 분위기 선택, 동반 산화·막질 영향 관리">분위기 제어</span> →
  <span title="Rs·SIMS·접합 누설 등으로 결과 검증 및 레시피 피드백">검사/계측</span>
""", unsafe_allow_html=True)
st.markdown("- 핵심 지표: 접합 깊이 xj, 표면 농도 Cs, 총 도즈 Q, 시트저항 Rs, 프로파일 균일도, 활성화/결함")

# 프로세스(가로 스크롤 카드)
st.subheader("프로세스(가로 스크롤 카드)")
steps = ["전처리 세정", "프리디포지션", "드라이브인", "분위기 제어", "검사/계측"]

html = """
<div style="overflow-x:auto; padding:6px 0;">
  <div style="display:flex; gap:12px; align-items:center; min-height:96px;">{items}</div>
</div>
"""
chip = """
<div style="flex:0 0 auto; min-width:200px; max-width:260px;
            padding:12px 14px; border:1.5px solid #d0d4dc; border-radius:14px;
            box-shadow:0 1px 3px rgba(0,0,0,0.06); background:#fff; text-align:center;">
  <div style="font-size:14px; font-weight:600;">{label}</div>
</div>
<div style="flex:0 0 auto; font-size:20px; margin:0 2px;">➜</div>
"""
items = "".join(
    (chip.format(label=s) if i < len(steps)-1 else chip.replace(
        '<div style="flex:0 0 auto; font-size:20px; margin:0 2px;">➜</div>', ''
    ).format(label=s))
    for i, s in enumerate(steps)
)
components.html(html.format(items=items), height=120, scrolling=False)

# 공정 단계 설명
st.subheader("공정 단계 설명 및 진도 관리")

#단계별 정보
steps_data = [
{
        "name": "전처리 세정",
        "icon": "🧼",
        "desc": """
🧼 **전처리 세정(Pre-clean)**

****왜 세정이 중요한가?****
- 표면 유기물/금속 이온/파티클/수분 제거 → 초기 도핑 균일성, 접합 누설 저감.
- 네이티브 산화막 상태 통일 → 표면 반응/주입 효율 재현성 향상.

**세정 방법**
- SC-1/SC-2로 유기물·금속 제거 → DIW 린스.
- 필요 시 HF-last(희박 HF)로 얇은 산화막 제거 후 **신속 로딩**(재산화 억제).
- 드라이(스핀/IPA/베이크)로 수분 제거.

**주의사항**
- HF 과공정 시 표면 거칠기 증가·디펙트 유발, HF-last 이후 공정 지연 금지.
- 세정제/DIW 품질(저항) 관리, 건조 얼룩/워터마크 방지.

🎯 **핵심 요약**
- ‘HF-last + 빠른 로딩’과 **재오염 방지**가 균일 확산의 출발점.
"""
    },
    {
        "name": "프리디포지션",
        "icon": "🧪",
        "desc": """
🧪 **프리디포지션(Predeposition)**

**개념**
- 표면에 도펀트 **표면 농도(Cs)** 또는 **총 도즈(Q)**를 부여하는 초기 단계.
- **무제한 소스(Constant Source)**: 표면에 도펀트가 충분 → 표면 농도 ≈ 용해한계에 근접.
- **제한 소스(Limited Source)**: 일정 도즈만 공급 → 전체 Q가 제한.

**소스 유형**
- 고체: BSG/PSG 글라스, 도펀트 코팅 웨이퍼 등.
- 액체: BBr₃, POCl₃ 등 소스 가열·반응 후 유리층 형성.
- 기체: PH₃, B₂H₆ 등(안전/독성 관리 필수).

**파라미터**
- 온도·시간·분압(소스 유량)·표면 반응 속도.
- 글라스 층 두께/조성(BSG/PSG) → 이후 제거 공정 포함.

**주의사항**
- **동반 산화**로 표면 재산화/두께 증가 → 후속 드라이브인 조건에 반영.
- 도펀트 교차오염(챔버 메모리) 방지: 프리컨디셔닝·더미 웨이퍼 사용.
- 소스 취급 안전(독성/가연성), 배기/스컴 제거.

🎯 **핵심 요약**
- 목적에 따라 **무제한/제한 소스** 전략을 선택, Cs 또는 Q를 안정적으로 부여.
"""
    },
    {
        "name": "드라이브인",
        "icon": "🔥",
        "desc": """
🔥 **드라이브인(Drive-in)**

**메커니즘**
- 고온(보통 900~1100℃)에서 Fick 법칙에 따라 도펀트가 벌크로 확산 → **접합 깊이(xj)** 형성.
- 무제한 소스: 표면 농도 **거의 일정** → 에르프컴플(Erfc)형 프로파일.
- 제한 소스: 총 도즈 **고정** → 가우시안형 프로파일.

**주요 파라미터**
- 온도·시간(열 예산), 불활성/산소 분위기, 슬립/워프 방지 램프 제어.
- 도펀트 종별 확산 계수(D) 차이(B > P > As 경향), 결정 방향/결함 영향.

**프로파일/전기특성**
- xj, Rs(시트저항), 표면 농도 감소/플랫닝 관리.
- 과도한 드라이브인은 옆 방향 확산(래터럴) 및 접합 얕음/깊음 불일치 유발.

**주의사항**
- 이미 형성된 금속/박막이 있을 경우 **열 윈도우** 준수(확산/상변화/박리).
- 산소 존재 시 추가 산화 발생 → 두께·전기 특성 변동.
- 슬립 라인/워프/크랙 등 기계적 결함 모니터링.

🎯 **핵심 요약**
- **열 예산(온도×시간)**을 정밀 제어해 **목표 xj와 Rs**를 동시에 만족.
"""
    },
    {
        "name": "분위기 제어",
        "icon": "🌫️",
        "desc": """
🌫️ **분위기 제어(Ambient Control)**

**선택지**
- **불활성(N₂/Ar)**: 순수 확산에 집중, 막질 변화 최소화.
- **산소(O₂/스팀)**: 동반 산화로 표면 상태 안정화/글라스층 형성·성숙화(상황에 따라 선택).

**장비/조건**
- 퍼니스(수평/수직) 또는 RTA(스파이크/쇼트 어닐).
- 분압·유량·램프/소크 제어, 퍼지로 잔류가스 제거.

**주의사항**
- 산소 혼입 시 예기치 않은 산화 두께 증가 → xj/Rs 변동.
- 라인 메모리·오염(교차 도핑) 방지 위한 **프리컨디셔닝/더미 로드**.
- 로딩 위치 의존성/웨이퍼 간 편차 관리.

🎯 **핵심 요약**
- **불활성 vs 산소** 목적을 명확히 하고, **분위기 안정화**로 재현성 확보.
"""
    },
    {
        "name": "검사/계측",
        "icon": "🔍",
        "desc": """
🔍 **검사/계측(Metrology)**

**전기/프로파일**
- **4-포인트 프로브(4PP)**로 Rs 측정(맵 측정으로 균일도 확인).
- **SIMS**로 농도 깊이 프로파일, **SRP**(Spreading Resistance)로 캐리어 프로파일.
- 접합 누설/브레이크다운, 다이오드 특성으로 전기적 건전성 평가.

**구조/표면**
- XTEM/HRXRD로 결정/결함, 표면 거칠기/산화막 변화 확인.
- BSG/PSG 잔류물, 글라스 제거 상태 점검(웨트 세정 후).

**주의사항**
- 측정 레시피(보정/모델) 일관성 유지, 고도핑 영역의 모델 선택 주의.
- 로트 간 캘리브레이션, 레퍼런스 웨이퍼 관리.

🎯 **핵심 요약**
- **Rs + SIMS**를 기준 지표로 삼고, 전기·구조 결과를 **레시피에 즉시 피드백**.
"""
    }]


# 페이지 진도 버킷
PAGE_PROGRESS_KEY = f"{CATEGORY_NAME}_progress"
# 과거 progress 키 제거
st.session_state.pop("progress", None)

if PAGE_PROGRESS_KEY not in st.session_state:
    st.session_state[PAGE_PROGRESS_KEY] = {s["name"]: False for s in steps_data}
else:
    for s in steps_data:
        st.session_state[PAGE_PROGRESS_KEY].setdefault(s["name"], False)

# 단계별 설명 및 체크박스
completed = 0
for s in steps_data:
    with st.expander(f"{s['icon']} {s['name']}"):
        st.write(s["desc"])
        checked = st.checkbox(
            "이 단계 학습 완료",
            value=st.session_state[PAGE_PROGRESS_KEY].get(s["name"], False),
            key=f"{CATEGORY_NAME}_{s['name']}"  # ← 네임스페이스로 키 충돌 방지
        )
        st.session_state[PAGE_PROGRESS_KEY][s["name"]] = checked
        if checked:
            completed += 1

# 전체 진도율 표시
total = len(steps_data)
percent = int((completed / total) * 100)
st.progress(percent)
st.caption(f"📘 학습 진도: {completed} / {total} 단계 완료 ({percent}%)")
# 질의응답
st.subheader("질의응답")

# 질의응답 상단 툴바: 대화 초기화 버튼
c1, c2 = st.columns([1, 9])
with c1:
    if st.button("대화 초기화", key="btn_clear_qa", use_container_width=True, help="질의응답 대화 내용 전체 삭제"):
        # 대화 이력 비우기
        st.session_state["chat_history"] = []



        st.toast("질의응답 대화가 초기화되었습니다. 🧹")
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()


# PDF 자료 넣기
if "vectorstore" not in st.session_state:
    st.info("임베딩 자료가 없습니다. 메인에서 PDF 업로드 → 임베딩 생성 후 이용하세요.")
else:
    if "qa_chain" not in st.session_state:
        # ⬇️ LLM.py의 함수로 백엔드/모델/LLM을 가져옵니다.
        try:
            backend, model = get_llm_backend()   # "openai" | "gemini", 모델 문자열
        except Exception:
            backend = st.session_state.get("llm_backend", "openai")
            model   = st.session_state.get("llm_model", "gpt-4o-mini")

        try:
            llm = get_chat_llm(backend=backend, model=model, temperature=0.2)
        except Exception as e:
            st.error(f"LLM 초기화 실패: {e}")
            llm = None

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

        # PDF우선 PDF에 내용 없으면 LLM이 알아서 답변해줌
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.session_state.qa_mode = "manual"   # 기본 수동, CRC 되면 "crc"로 변경

        if llm is not None:
            try:
                # ▼ CRC 시도 (이전 대화 맥락을 직접 넘길 수 있음)
                from langchain.chains import ConversationalRetrievalChain
                from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "당신의 1차 정보원은 업로드된 PDF입니다. "
                     "가능하면 PDF 근거를 우선하여 답하고, 부족하면 일반지식으로 보완하되 그 사실을 한 문장으로 표시하십시오. "
                     "항상 정중한 한국어(존댓말)로 답하십시오."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])

                st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": prompt},
                )
                st.session_state.qa_mode = "crc"
            except Exception:
                # 정보 없으면 LLM 알아서 답변
                st.session_state.qa_chain = None

    # 채팅 내역 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # [{'role':'user'|'assistant', 'content': str, 'sources': list}]

    # 과거 채팅 표시 (최근 2개만 기본, 나머지는 펼쳐서 보기)
    if st.session_state.chat_history:
        older = st.session_state.chat_history[:-2]
        recent = st.session_state.chat_history[-2:]

        if older:
            with st.expander(f"이전 대화 보기 ({len(older)}개)", expanded=False):
                for msg in older:
                    with st.chat_message("user" if msg["role"]=="user" else "assistant"):
                        st.markdown(msg["content"])
                        if msg.get("sources"):
                            with st.popover("출처 보기"):
                                for i, meta in enumerate(msg["sources"], 1):
                                    st.caption(f"{i}. {meta}")

        for msg in recent:
            with st.chat_message("user" if msg["role"]=="user" else "assistant"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.popover("출처 보기"):
                        for i, meta in enumerate(msg["sources"], 1):
                            st.caption(f"{i}. {meta}")

    # 입력창
    with st.form("qa_form", clear_on_submit=True):
        user_q = st.text_input("질문을 입력하세요… (예: EUV와 DUV 차이)", key="qa_text")
        submitted = st.form_submit_button("Send")

    # 버튼을 눌렀고 비어 있지 않을 때만 생성
    if submitted and user_q and user_q.strip():
        # 1) 사용자 메시지 기록 & 표시
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        # 2) 응답 생성
        if st.session_state.get("qa_mode") == "crc" and st.session_state.get("qa_chain") is not None:
            # CRC 경로: 대화 맥락을 chat_history 인자로 직접 전달
            with st.chat_message("assistant"):
                with st.status("검색 및 응답 생성 중...", expanded=False):
                    out = st.session_state.qa_chain({
                        "question": user_q,
                        "chat_history": hist_pairs(st.session_state.chat_history, limit_pairs=6)  # ← LLM.py 함수
                    })

                answer = out.get("answer") or out.get("result") or "정보가 부족합니다"
                st.markdown(answer)

                # 출처 요약
                srcs = []
                for sdoc in (out.get("source_documents") or []):
                    meta = sdoc.metadata or {}
                    srcs.append(f"{meta.get('source','파일')} p.{meta.get('page','?')}")
                if srcs:
                    with st.popover("출처 보기"):
                        for i, meta in enumerate(srcs, 1):
                            st.caption(f"{i}. {meta}")
                # 히스토리 저장
                st.session_state.chat_history.append({"role":"assistant", "content":answer, "sources":srcs})

        else:
            # 수동 RAG 폴백: 문서 검색 + 대화맥락/발췌를 직접 프롬프트에 주입
            with st.chat_message("assistant"):
                with st.status("검색 및 응답 생성 중...", expanded=False):
                    llm = st.session_state.get("llm", None)
                    retriever = st.session_state.get("retriever", None)

                    docs = []
                    if retriever is not None:
                        try:
                            docs = retriever.get_relevant_documents(user_q)
                        except Exception:
                            docs = []

                    hist_txt = hist_text(st.session_state.chat_history, limit_pairs=6)  # ← LLM.py 함수
                    doc_block = summarize_docs(docs, max_chars=2400)                  # ← LLM.py 함수

                    if doc_block:
                        prompt = (
                            "규칙:\n"
                            "1) 아래 [PDF 발췌]에서 먼저 근거를 찾고 답하십시오.\n"
                            "2) 충분한 근거가 없으면 일반지식으로 보완하고, 그 사실을 한 문장으로 표시하십시오.\n"
                            "3) 한국어(존댓말)로 간결하고 정확히 답하십시오.\n\n"
                            f"[대화 맥락]\n{hist_txt or '(이전 대화 없음)'}\n\n"
                            f"[PDF 발췌]\n{doc_block}\n\n"
                            f"[질문]\n{user_q}\n"
                        )
                    else:
                        prompt = (
                            "다음은 최근 대화입니다.\n"
                            f"{hist_txt or '(이전 대화 없음)'}\n\n"
                            "업로드된 PDF에서 충분한 근거를 찾지 못했습니다. 일반지식으로 답하되, "
                            "모호하면 '정보가 부족합니다'라고 밝혀주십시오. 한국어(존댓말)로 답하십시오.\n"
                            f"[질문] {user_q}"
                        )

                    if llm is None:
                        answer = "정확히 알 수 없습니다."
                    else:
                        try:
                            out = llm.invoke(prompt)
                            answer = getattr(out, "content", None) or getattr(out, "text", None) or "정보가 부족합니다"
                        except Exception:
                            answer = "정확히 알 수 없습니다."

                st.markdown(answer)

                # 출처 요약
                srcs = []
                for sdoc in (docs or []):
                    meta = getattr(sdoc, "metadata", {}) or {}
                    srcs.append(f"{meta.get('source','파일')} p.{meta.get('page','?')}")
                if srcs:
                    with st.popover("출처 보기"):
                        for i, meta in enumerate(srcs, 1):
                            st.caption(f"{i}. {meta}")

                # 히스토리 저장
                st.session_state.chat_history.append({"role":"assistant", "content":answer, "sources":srcs})


# 랜덤 문제 생성기 + 채점
st.subheader("랜덤 문제 생성기")
CATEGORY_NAME = "포토리소그래피"  # ← 페이지 주제명

# (중복 회피용 )
hist_key = f"{CATEGORY_NAME}_quiz_history"
if hist_key not in st.session_state:
    st.session_state[hist_key] = []  # 문자열(서술형 질문) 또는 MC 질문 텍스트 저장

# 설정
cols = st.columns(3)
difficulty   = cols[0].selectbox(
    "난이도",
    ["초급", "고급"],
    index=0,
    key=f"{CATEGORY_NAME}_difficulty"
)
n_items      = cols[1].selectbox(
    "문항 수",
    [1, 3, 5],
    index=1,
    key=f"{CATEGORY_NAME}_n_items"
)
has_vs       = "vectorstore" in st.session_state
with_context = cols[2].checkbox(
    "업로드 문서 기반(권장)",
    has_vs,
    key=f"{CATEGORY_NAME}_with_context"
)

# 프롬프트
QUIZ_PROMPT_MC = """\
당신은 반도체 공정 과목의 교수입니다.
주제: {category}
난이도: 초급
출제 문항 수: {n_items}

{context}

요구사항:
- 4지선다 객관식 문제를 {n_items}개 생성
- 각 문항은 반드시 아래 '정확한 형식'을 지킬 것 (추가 텍스트 금지)
- 보기는 A) B) C) D) 로 표시, 정답은 하나만
- 각 문항에 간단한 해설 1~2문장 포함

[정확한 형식 예시 — 이 틀을 그대로 지킬 것]
1) 질문 텍스트
A) 보기 A
B) 보기 B
C) 보기 C
D) 보기 D
정답: A
해설: 한두 문장 설명
"""

QUIZ_PROMPT_TXT = """\
당신은 반도체 공정 과목의 교수입니다.
주제: {category}
난이도: 고급
출제 문항 수: {n_items}

{context}

위 내용을 참고하여, 주제에 맞는 랜덤 서술형 문제를 {n_items}개 만들어주세요.
문항은 1), 2), 3)... 처럼 번호를 붙여 한 줄씩 시작하세요.
답은 포함하지 마세요.
"""

EVAL_PROMPT_TMPL = """\
당신은 {category} 분야의 채점 보조입니다.
다음 문항과 수험자 답안을 평가하세요.

[문항]
{question}

[수험자 답안]
{answer}

(선택) 참고 컨텍스트:
{context}

평가 기준:
- 사실 일치 여부, 핵심 개념 포함 여부, 논리성.
- 간결히 '정답' 또는 '오답'으로 판정하고, 2~3문장의 피드백 제공.

반드시 아래 형식을 정확히 지키세요(줄바꿈 포함, 다른 텍스트 금지):
판정: 정답|오답
피드백: <두세 문장 피드백>
"""

# 문제 생성 버튼 동작
if st.button("랜덤 문제 생성", use_container_width=True):
    ph = st.empty()
    with ph.container():
        if hasattr(st, "status"):
            with st.status("문제 생성 중...", expanded=True) as status:
                status.update(label="컨텍스트 수집...", state="running")
                backend, model = get_llm_backend()
                context = gather_context(k=6, enabled=with_context, retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6}) if has_vs else None)

                status.update(label="프롬프트 구성...", state="running")
                if difficulty == "초급":
                    prompt = QUIZ_PROMPT_MC.format(
                        category=CATEGORY_NAME,
                        n_items=n_items,
                        context=(f"[컨텍스트]\n{context}" if context else "(컨텍스트 없음)")
                    )
                else:
                    prompt = QUIZ_PROMPT_TXT.format(
                        category=CATEGORY_NAME,
                        n_items=n_items,
                        context=(f"[컨텍스트]\n{context}" if context else "(컨텍스트 없음)")
                    )

                status.update(label="문항 생성 요청...", state="running")
                raw = generate_with_openai(prompt, model) if backend == "openai" else generate_with_gemini(prompt, model)

                prev_texts = [p if isinstance(p, str) else p.get("q","") for p in st.session_state[hist_key]]

                if difficulty == "초급":
                    cand = parse_mc_questions(raw, n_items)
                    uniques = []
                    for item in cand:
                        if not any(is_similar(item["q"], pt) for pt in prev_texts):
                            uniques.append(item)
                    if len(uniques) < n_items:
                        need = n_items - len(uniques)
                        status.update(label=f"보강 생성 ({need}개)...", state="running")
                        raw2 = generate_with_openai(prompt, model) if backend == "openai" else generate_with_gemini(prompt, model)
                        cand2 = parse_mc_questions(raw2, need)
                        for it in cand2:
                            if len(uniques) >= n_items: break
                            if not any(is_similar(it["q"], pt) for pt in (prev_texts + [u["q"] for u in uniques])):
                                uniques.append(it)

                    st.session_state[f"{CATEGORY_NAME}_quiz_items"] = uniques
                    st.session_state[f"{CATEGORY_NAME}_quiz_mode"]  = "초급"
                    st.session_state[hist_key].extend([u["q"] for u in uniques])

                else:  # 고급(서술형)
                    cand = extract_questions(raw, n_items)
                    uniques = []
                    for q in cand:
                        if not any(is_similar(q, pt) for pt in prev_texts):
                            uniques.append(q)
                    if len(uniques) < n_items:
                        need = n_items - len(uniques)
                        status.update(label=f"보강 생성 ({need}개)...", state="running")
                        raw2 = generate_with_openai(prompt, model) if backend == "openai" else generate_with_gemini(prompt, model)
                        cand2 = extract_questions(raw2, need)
                        for q in cand2:
                            if len(uniques) >= n_items: break
                            if not any(is_similar(q, pt) for pt in (prev_texts + uniques)):
                                uniques.append(q)

                    st.session_state[f"{CATEGORY_NAME}_quiz_items"] = uniques
                    st.session_state[f"{CATEGORY_NAME}_quiz_mode"]  = "고급"
                    st.session_state[hist_key].extend(uniques)

                status.update(label="완료 ✅", state="complete")
        else:
            bar = st.progress(0)
            backend, model = get_llm_backend(); bar.progress(10)
            context = gather_context(k=6, enabled=with_context, retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6}) if has_vs else None); bar.progress(20)
            if difficulty == "초급":
                prompt = QUIZ_PROMPT_MC.format(category=CATEGORY_NAME, n_items=n_items, context=(f"[컨텍스트]\n{context}" if context else "(컨텍스트 없음)"))
            else:
                prompt = QUIZ_PROMPT_TXT.format(category=CATEGORY_NAME, n_items=n_items, context=(f"[컨텍스트]\n{context}" if context else "(컨텍스트 없음)"))
            raw = generate_with_openai(prompt, model) if backend == "openai" else generate_with_gemini(prompt, model); bar.progress(60)
            prev_texts = [p if isinstance(p, str) else p.get("q","") for p in st.session_state[hist_key]]
            if difficulty == "초급":
                cand = parse_mc_questions(raw, n_items)
                uniques = [it for it in cand if not any(is_similar(it["q"], pt) for pt in prev_texts)]
                st.session_state[f"{CATEGORY_NAME}_quiz_items"] = uniques
                st.session_state[f"{CATEGORY_NAME}_quiz_mode"]  = "초급"
                st.session_state[hist_key].extend([u["q"] for u in uniques])
            else:
                cand = extract_questions(raw, n_items)
                uniques = [q for q in cand if not any(is_similar(q, pt) for pt in prev_texts)]
                st.session_state[f"{CATEGORY_NAME}_quiz_items"] = uniques
                st.session_state[f"{CATEGORY_NAME}_quiz_mode"]  = "고급"
                st.session_state[hist_key].extend(uniques)
            bar.progress(100)
    ph.empty()

#  문제 표시  답안 입력 채점
items = st.session_state.get(f"{CATEGORY_NAME}_quiz_items", [])
mode  = st.session_state.get(f"{CATEGORY_NAME}_quiz_mode", "고급")

if items:
    st.markdown("### 생성된 문제")

    if mode == "초급":
        # 객관식
        for i, it in enumerate(items, start=1):
            st.markdown(f"**{i}) {it['q']}**")
            key = f"{CATEGORY_NAME}_mc_{i-1}"
            choice = st.radio("보기 선택", options=it["opts"], key=key, index=None)
            st.caption("정답 선택 후 아래 '채점하기'를 누르세요.")
        if st.button("채점하기", type="primary", use_container_width=True):
            st.markdown("### 채점 결과")
            for i, it in enumerate(items, start=1):
                key = f"{CATEGORY_NAME}_mc_{i-1}"
                sel = st.session_state.get(key)
                sel_letter = sel.split(")")[0] if sel else None
                correct = (sel_letter == it["answer"])
                verdict = "정답" if correct else "오답"
                st.markdown(f"**문항 #{i} 결과**")
                st.markdown(f"**판정: {verdict}**")
                st.markdown(f"피드백: {it.get('expl','(해설 없음)')}")
                st.markdown("---")
    else:
        # 서술형
        for i, qtext in enumerate(items, start=1):
            st.markdown(f"**{i}) {qtext}**")
            st.text_area(
                f"답안 입력 #{i}",
                key=f"{CATEGORY_NAME}_ans_{i-1}",
                height=100,
                placeholder="여기에 본인 답안을 작성하세요."
            )
#채점
        if st.button("채점하기", type="primary", use_container_width=True):
            backend, model = get_llm_backend()
            context = gather_context(k=6, enabled=with_context, retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6}) if has_vs else None)
            results = []
            for i, qtext in enumerate(items):
                ans = st.session_state.get(f"{CATEGORY_NAME}_ans_{i}", "").strip()
                eval_prompt = EVAL_PROMPT_TMPL.format(
                    category=CATEGORY_NAME,
                    question=qtext,
                    answer=ans if ans else "(무응답)",
                    context=(f"[컨텍스트]\n{context}" if context else "(컨텍스트 없음)")
                )
                judged = generate_with_openai(eval_prompt, model) if backend == "openai" else generate_with_gemini(eval_prompt, model)
                results.append(judged)

            st.markdown("### 채점 결과")
            for i, judged in enumerate(results, start=1):
                st.markdown(f"**문항 #{i} 결과**")
                v, fb = parse_eval(judged)
                st.markdown(f"**판정: {v or '판정 불명'}**")
                st.markdown(f"피드백: {fb or '(없음)'}")
                st.markdown("---")
else:
    st.caption("아직 생성된 문제가 없습니다. ‘랜덤 문제 생성’을 눌러주세요.")
