import os
import streamlit as st
import streamlit.components.v1 as components

from LLM import (
    is_similar,
    get_llm_backend,
    generate_with_openai,
    generate_with_gemini,
    gather_context,
    extract_questions,
    parse_mc_questions,
    parse_eval,
    get_chat_llm,
    get_llm_backend,
    get_chat_llm,
    hist_pairs,
    hist_text,
    summarize_docs
)


st.set_page_config(page_title="증착", layout="wide")

st.header("6) 증착 (CVD/PVD/ALD)")

CATEGORY_NAME = "증착"

st.subheader("개요")
st.write(
    "웨이퍼 표면에 원하는 재료의 박막을 형성하는 공정입니다. 물리적 증착(PVD: 스퍼터/증발), "
    "화학적 증착(CVD: LPCVD/PECVD/RTCVD), 원자층 증착(ALD)을 사용하며, "
    "조성·두께·균일도·응력·콘포멀리티·결함/파티클 관리가 핵심입니다."
)

# 핵심 포인트 (툴팁)
st.subheader("핵심 포인트")
st.markdown("""
- <span title="오염 제거·표면 활성화·접착 향상·핵생성 안정화">전처리/시드</span> →
  <span title="PVD/CVD/ALD 중 목적에 맞는 방식 선택·레시피 설정">막 형성</span> →
  <span title="두께·조성·응력·입자 최소화를 위한 파라미터 최적화">조건 최적화</span> →
  <span title="어닐/플라즈마 등으로 막질 개선·결합/밀도 향상">후처리</span> →
  <span title="두께·조성·응력·결함/파티클 정량화 및 피드백">검사/계측</span>
""", unsafe_allow_html=True)
st.markdown("- 핵심 지표: 두께/균일도, 막질(밀도/조성/결합), 응력(인장/압축), 콘포멀리티/스텝 커버리지, 파티클/디펙트, 표면 거칠기(Ra/RMS)")

# 프로세스(가로 스크롤 카드)
st.subheader("프로세스(가로 스크롤 카드)")
steps = ["전처리/시드", "막 형성(PVD/CVD/ALD)", "조건 최적화", "후처리", "검사/계측"]

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

# 공정 단계 설명 + 진도 관리
st.subheader("공정 단계 설명 및 진도 관리")

steps_data = [
    {
        "name": "전처리/시드",
        "icon": "🧼",
        "desc": """
🧼 **전처리/시드(Pre-treatment / Seed)**

**무엇을/왜**
- 유기물/금속 이온/수분/파티클 제거, 표면 활성화로 **접착력**과 **핵생성 안정성** 확보.
- 콘포멀 증착이 어려운 구조에서는 **시드층**으로 초기 연속성 제공(특히 금속 도금/ECP 전 단계).

**어떻게**
- 습식: SC-1/SC-2, DHF(HF-last)로 네이티브 산화막 제어 → DIW 린스·건조.
- 드라이: Ar 플라즈마 클린, H₂/N₂ 플라즈마 리덕션, UV/오존 유기 제거.
- 접착 향상: HMDS/SAM, 프라이머, 플라즈마 활성화.
- 시드: PVD/ALD로 수~수십 nm 수준 균일/연속 막 형성.

**주의사항**
- DHF 과공정 → 표면 거칠기↑/미세 패턴 손상. HF-last 후 **지연 최소화**.
- 플라즈마/UV 과노출 → 표면 데미지/과도한 친수화로 재오염 리스크.
- 시드 불연속/섬형 성장 → 이후 막의 공극/시임 원인.

🎯 **핵심 요약**
- **청정+활성화+적절한 시드**가 증착 막질·접착의 출발점.
"""
    },
    {
        "name": "막 형성 (PVD/CVD/ALD)",
        "icon": "🛠️",
        "desc": """
🛠️ **막 형성(Deposition) — PVD/CVD/ALD 선택**

**PVD (Physical Vapor Deposition)**
- 스퍼터/증발 기반 **라인-오브-사이트** 우세, 빠른 증착, 합금/금속 박막에 유리.
- 매개변수: 타깃 전력, 압력(Ar), 바이어스, 기판 온도, 거리, 마그네트론/히터.
- 장점: 높은 순도/생산성, 쉬운 조성 제어. 단점: 콘포멀리티/하이 AR 충전 취약.

**CVD (Chemical Vapor Deposition)**
- 기상 전구체 반응으로 필름 성장(LPCVD/PECVD/RTCVD).
- 매개변수: 온도, 압력, 전구체/캐리어 유량, RF 파워(PECVD), 반응/폐기 라인 온도.
- 장점: 스텝 커버리지↑, 대면적 균일도↑. 단점: 불순물/하이드로겐/저밀도 위험(특히 저온 PECVD).

**ALD (Atomic Layer Deposition)**
- **자기 제한 반응** 기반 **사이클식** 증착: A 퍼지 B 퍼지 … (표면 포화).
- 매개변수: 사이클 수(=두께), 펄스/퍼지 시간, 기판 온도(윈도우), 전구체 반응성.
- 장점: **탁월한 콘포멀리티**·원자층 제어·핀홀 최소. 단점: 속도↓, 전구체 비용/독성 관리.

**주의사항**
- 전구체 순도/라인 컨디션 불량 → 파티클/폴리머 잔류/핵형성 지연.
- PVD의 재비산/섀도잉, CVD의 기상 폴리머, ALD의 반응 미포화(언더퍼지) 주의.
- 하부막/온도 윈도우/플라즈마 데미지/차징 등 디바이스 제약 고려.

🎯 **핵심 요약**
- **PVD=금속 빠름**, **CVD=커버리지/생산성**, **ALD=콘포멀·정밀** — 목적에 맞춰 선택.
"""
    },
    {
        "name": "조건 최적화",
        "icon": "⚙️",
        "desc": """
⚙️ **조건 최적화(Process Tuning)**

**목표**
- 두께·조성·응력(인장/압축)·밀도·표면 거칠기·결함/파티클을 **동시에** 만족.

**파라미터 예시**
- 온도/압력/유량/전력(RF/DC)/플라즈마 듀티/바이어스.
- PVD: 파워·압력·기판 바이어스·기판 회전·각도 증착(리니어라이즈).
- CVD/ALD: 전구체/코리액턴트 펄스·퍼지·사이클, 라인 온도, 챔버 컨디셔닝.

**응력/막질 제어**
- 온도/전력/압력 조합으로 응력 제어(보우/커브 측정 피드백).
- ALD 사이클 파라미터로 밀도/결합 상태 튜닝(예: 산화물 vs 질화물).
- 하부막과의 계면 반응/갈바닉/디퓨전 고려(배리어 필요 시 별도 단계 추가).

**주의사항**
- 과도한 플라즈마 파워 → 표면 손상/차징/트랩 증가.
- 퍼지 부족 → 전구체 크로스리액션으로 파티클/폴리머 발생.
- 레시피 변경 시 **챔버 메모리**/더미 웨이퍼 컨디셔닝 필수.

🎯 **핵심 요약**
- **응력–밀도–커버리지** 트레이드오프를 지표(두께/응력/결함)로 닫는 **실험계획(DoE)**가 관건.
"""
    },
    {
        "name": "후처리",
        "icon": "🔥",
        "desc": """
🔥 **후처리(Post-treatment)**

**목적**
- 결합/밀도 향상, 잔류 라디칼/리간드 제거, 전기·기계 특성 개선, 접촉저항 저감.

**방법**
- 어닐: N₂/Ar/H₂(포밍가스) RTA/퍼니스 — 수소 패시베이션, 밀도↑.
- 플라즈마: H₂/N₂/O₂/Ar 플라즈마로 표면 결합/오염 제거, 표면 에너지 조절.
- UV/오존: 유기 잔류 제거, 친수성/소수성 조절.

**주의사항**
- 메탈/유전막의 **열 안정성 윈도우** 준수(상변화/디퓨전/박리).
- 과도한 H₂ 플라즈마/고온 어닐 → 박막/계면 열화·균열·스트레스 상승.
- 후처리로 응력 변동 가능 → CMP/포토 정렬에 영향.

🎯 **핵심 요약**
- **어닐/플라즈마**로 막질을 끌어올리되, **열·플라즈마 윈도우** 내에서 운전.
"""
    },
    {
        "name": "검사/계측",
        "icon": "🔍",
        "desc": """
🔍 **검사/계측(Metrology)**

**두께/형상**
- 타원계/스펙트럼 반사율/프로파일러, XRR로 두께·밀도·조도.
- 콘포멀리티/스텝 커버리지: 크로스섹션 SEM/TEM, FIB 단면.

**조성/결합**
- XPS/ToF-SIMS/FTIR로 조성/결합 상태, 불순물( C/H/N/O ) 정량.
- XRD로 결정상/응력, 라만으로 응력/결합 확인.

**결함/파티클/응력**
- 파티클 스캐너/검사, 표면 거칠기(AFM), 보우/커브로 응력 측정.
- 전기적: 누설/유전 손실/저항, 콘택 저항 Kelvin 구조.

**주의사항**
- 레퍼런스 웨이퍼/측정 레시피 고정으로 로트 간 비교성 확보.
- 박막/고 AR 구조 측정은 **모델 의존성** 크므로 상호 보완 측정 권장.

🎯 **핵심 요약**
- **두께·조성·응력·결함**을 다각도로 계측하고, 결과를 즉시 레시피에 **피드백**.
"""
    },
]

# === 페이지 전용 진도 버킷 ===
PAGE_PROGRESS_KEY = f"{CATEGORY_NAME}_progress"
# (이전 전역 progress 키 제거하여 충돌 방지)
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
            key=f"{CATEGORY_NAME}_{s['name']}"
        )
        st.session_state[PAGE_PROGRESS_KEY][s["name"]] = checked
        if checked:
            completed += 1

# 전체 진도율
total = len(steps_data)
percent = int((completed / total) * 100)
st.progress(percent)
st.caption(f"📘 학습 진도: {completed} / {total} 단계 완료 ({percent}%)")

# ---------------- 질의응답 (RAG · 챗봇 UI · LLM.py 함수 사용) ----------------
st.subheader("질의응답")

# ── 질의응답 상단 툴바: 대화 초기화 버튼
c1, c2 = st.columns([1, 9])
with c1:
    if st.button("대화 초기화", key="btn_clear_qa", use_container_width=True, help="질의응답 대화 내용 전체 삭제"):
        # 대화 이력 비우기
        st.session_state["chat_history"] = []

        # (선택) 체인 재생성을 원하시면 아래 주석을 해제
        # st.session_state.pop("qa_chain", None)
        # st.session_state.pop("qa_mode", None)

        # (선택) 요약 메모리 등을 쓰신다면 함께 초기화
        # st.session_state.pop("MEMORY", None)

        st.toast("질의응답 대화가 초기화되었습니다. 🧹")
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()


# 체인 준비
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

        # 메모리 지원 체인 구성: CRC 우선, 실패 시 수동 RAG 폴백용 상태 저장
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
                # CRC 불가(비호환 LLM 등) → 수동 RAG로 처리
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

    # ===== 입력창 =====
    with st.form("qa_form", clear_on_submit=True):
        user_q = st.text_input("질문을 입력하세요… (예: EUV와 DUV 차이)", key="qa_text")
        submitted = st.form_submit_button("Send")

    # ✅ 버튼을 눌렀고 비어 있지 않을 때만 생성
    if submitted and user_q and user_q.strip():
        # 1) 사용자 메시지 기록 & 표시
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        # 2) 응답 생성
        if st.session_state.get("qa_mode") == "crc" and st.session_state.get("qa_chain") is not None:
            # ---- CRC 경로: 대화 맥락을 chat_history 인자로 직접 전달
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
                # 히스토리 저장(출처 포함)
                st.session_state.chat_history.append({"role":"assistant", "content":answer, "sources":srcs})

        else:
            # ---- 수동 RAG 폴백: 문서 검색 + 대화맥락/발췌를 직접 프롬프트에 주입
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

                # 출처 요약(수동 경로)
                srcs = []
                for sdoc in (docs or []):
                    meta = getattr(sdoc, "metadata", {}) or {}
                    srcs.append(f"{meta.get('source','파일')} p.{meta.get('page','?')}")
                if srcs:
                    with st.popover("출처 보기"):
                        for i, meta in enumerate(srcs, 1):
                            st.caption(f"{i}. {meta}")

                # 히스토리 저장(출처 포함)
                st.session_state.chat_history.append({"role":"assistant", "content":answer, "sources":srcs})


# ---------------- 랜덤 문제 생성기 + 채점 ----------------
st.subheader("랜덤 문제 생성기")
CATEGORY_NAME = "포토리소그래피"  # ← 페이지 주제명

# (중복 회피용 히스토리)
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

# 프롬프트 템플릿 (변수)
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

# ----- 문제 표시 + 답안 입력 / 채점 -----
items = st.session_state.get(f"{CATEGORY_NAME}_quiz_items", [])
mode  = st.session_state.get(f"{CATEGORY_NAME}_quiz_mode", "고급")

if items:
    st.markdown("### 생성된 문제")

    if mode == "초급":
        # 객관식 렌더링
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
        # 서술형 렌더링 (LLM 채점)
        for i, qtext in enumerate(items, start=1):
            st.markdown(f"**{i}) {qtext}**")
            st.text_area(
                f"답안 입력 #{i}",
                key=f"{CATEGORY_NAME}_ans_{i-1}",
                height=100,
                placeholder="여기에 본인 답안을 작성하세요."
            )

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

