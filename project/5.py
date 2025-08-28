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

st.set_page_config(page_title="이온주입", layout="wide")

st.header("5) 이온주입 (Ion Implantation)")

CATEGORY_NAME = "이온주입"

st.subheader("개요")
st.write(
    "가속된 도펀트 이온(B, P, As 등)을 웨이퍼에 주입하여 원하는 깊이와 농도로 도핑하는 공정입니다. "
    "빔 에너지·도즈·빔 각도·채널링 억제·커런트/스캔 균일도 제어가 중요하며, 주입 후 활성화 어닐(RTA/스파이크)로 "
    "전기적 활성화 및 격자 결함을 회복합니다."
)

# 핵심 포인트 (툴팁)
st.subheader("핵심 포인트")
st.markdown("""
- <span title="표면 오염/수분 제거, PR/하드마스크 상태 점검">전처리 세정</span> →
  <span title="빔 에너지/도즈/스캔으로 목표 프로파일 형성">이온 주입</span> →
  <span title="틸트/트위스트로 채널링 억제, 오버레이 고려">각도/채널링 제어</span> →
  <span title="RTA/스파이크로 활성화·결함 회복, 확산 최소화">활성화 어닐</span> →
  <span title="Rs, SIMS, 결함/누설 평가로 레시피 피드백">검사/계측</span>
""", unsafe_allow_html=True)
st.markdown("- 핵심 지표: 도즈(ions/cm²), 투영거리 Rp/분산 ΔRp, 시트저항 Rs, 활성도(액티브 도핑), 표면/격자 결함, 균일도")

# 프로세스(가로 스크롤 카드)
st.subheader("프로세스(가로 스크롤 카드)")
steps = ["전처리 세정", "이온 주입", "각도/채널링 제어", "활성화 어닐", "검사/계측"]

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

# 단계별 정보
steps_data = [
    {
        "name": "전처리 세정",
        "icon": "🧼",
        "desc": """
🧼 **전처리 세정(Pre-clean)**

****왜 세정이 중요한가?****
- 표면 유기물·금속이온·파티클·수분 제거 → 전하 축적/마이크로 마스킹 방지, 주입 균일도 향상.
- PR/하드마스크 두께·내구성 확인 → 주입 중 스퍼터/데미지 대비.

**세정 방식**
- SC-1/SC-2 후 DIW 린스, 건조(스핀/IPA/베이크).
- 필요 시 PR 하드베이크로 내플라즈마성 향상, 또는 하드마스크( SiO₂/SiN ) 상태 점검.

**주의사항**
- 과세정/과건조는 PR 크랙·에지 리프트 유발 가능.
- 실리콘 표면의 수분/오염은 빔 전하 축적(차징)과 국부 과주입을 초래.

🎯 **핵심 요약**
- **청정·건조·마스크 상태**가 주입 프로파일 재현성의 출발점.
"""
    },
    {
        "name": "이온 주입",
        "icon": "🚀",
        "desc": """
🚀 **이온 주입(Ion Implantation)**

**핵심 파라미터**
- **에너지(keV~MeV)**: 평균 깊이(Rp) 결정, 고에너지일수록 깊이↑.
- **도즈(ions/cm²)**: 농도·Rs 결정, 고도즈 시 채널 플러깅/스퍼터/표면 거칠기 증가.
- **빔커런트/스캔**: 처리 속도·열적 부하·균일도(웨이퍼 내/간) 영향.
- **빔 스폿/스캔 패턴**: 에지/센터 균일도, 오버스캔 보정.

**장비/모드**
- 싱글 웨이퍼 스캐닝/스텝핑, 클러스터 통합(프리클린/어닐).
- 중/저/고 에너지 라인, 중성자 빔 필터·다중 전하 상태 관리.

**주의사항**
- 고도즈에서 PR/하드마스크 스퍼터 → 입자/커버리지 변화.
- 웨이퍼 온도 상승 → 도펀트 이동/데미지 누적, 쿨링/펄스 운전 고려.
- 금속 존재 시 재비산·오염 주의, 정전척(ESC) 클린 상태 확보.

🎯 **핵심 요약**
- **에너지=깊이, 도즈=농도**. 빔 커런트/스캔으로 **균일도와 열부하**를 함께 관리.
"""
    },
    {
        "name": "각도/채널링 제어",
        "icon": "📐",
        "desc": """
📐 **각도/채널링 제어(Angle / De-channeling)**

**개념**
- 결정 방향 채널을 따라 이온이 깊게 침투하는 **채널링** 현상 억제가 필수(프로파일 꼬리 방지).
- **틸트(기울임)**, **트위스트(회전)** 각도를 부여해 결정 채널과 정렬을 피함.

**실무 설정**
- 보통 **틸트 7°±**, **트위스트 20°±**(장비/공정마다 최적화)로 채널링 최소화.
- 마스크 에지 근방의 셰도우/리디포지션 고려, 오버레이/PR 슬로프 영향.

**주의사항**
- 과도한 각도는 **CD 변형/어깨 주입** 유발, 비아/트렌치 내부 주입 불균일.
- 웨이퍼 워프/ESC 클램핑 불량 시 각도 재현성 저하.

🎯 **핵심 요약**
- **적정 틸트·트위스트**로 채널링을 억제하되, **오버레이·지오메트리** 영향까지 함께 최적화.
"""
    },
    {
        "name": "활성화 어닐",
        "icon": "🔥",
        "desc": """
🔥 **활성화 어닐(Activation Anneal)**

**목적**
- 주입으로 발생한 **격자 결함 회복**, 도펀트의 **전기적 활성화**(치환형 자리 이동).

**방법**
- **RTA/스파이크 어닐**: 수십 ms~수 s 저열예산로 활성화↑ 확산↓.
- 퍼니스 어닐: 균일성 우수, 그러나 확산↑ → 노드/디바이스에 따라 선택.

**파라미터**
- 피크 온도, 램프 업/다운 속도, 유지 시간(소크), 분위기(N₂/Ar/H₂).
- 메탈 스택/유전막의 열 안정성 윈도우 준수.

**주의사항**
- 과도한 열 예산은 **접합 확산/쇼트 채널 영향**. 반대로 열 부족 시 활성도↓, Rs↑.
- H₂ 안전, 산화/환원 분위기 혼입으로 표면/계면 변성 주의.

🎯 **핵심 요약**
- **짧고 높은 온도(RTA/스파이크)**로 **활성 최대·확산 최소**의 균형을 달성.
"""
    },
    {
        "name": "검사/계측",
        "icon": "🔍",
        "desc": """
🔍 **검사/계측(Metrology)**

**전기/프로파일**
- **4-포인트 프로브(4PP)**로 Rs 맵 측정(웨이퍼 내/간 균일도).
- **SIMS**로 농도 깊이 프로파일(Rp, ΔRp), **SRP**로 캐리어 프로파일 확인.
- 다이오드/콘택 구조의 **누설/브레이크다운** 측정.

**결함/표면**
- TEM/XTEM로 결함/재결정화, AFM/SEM로 표면 거칠기·스퍼터 흔적 확인.
- XRD/라만으로 응력·결정성 평가.

**주의사항**
- 측정 레시피/보정 일관성 유지(특히 고도즈/고에너지).
- PR/글라스 잔류물 제거 후 측정, 측정 전 ESD/차징 방지.

🎯 **핵심 요약**
- **Rs + SIMS**를 핵심 지표로 삼고, 결함·누설을 함께 확인해 **레시피를 즉시 피드백**.
"""
    },
]

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
            key=f"{CATEGORY_NAME}_{s['name']}" # ← 네임스페이스로 키 충돌 방지
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

# ── 질의응답 상단 툴바: 대화 초기화 버튼
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
        # ⬇️ LLM.py의 함수로 백엔드/모델/LLM을 가져옴
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
        st.session_state.qa_mode = "manual"  # 기본 수동이지만 CRC 되면 "crc"로 변경

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
                # 히스토리 저장
                st.session_state.chat_history.append({"role":"assistant", "content":answer, "sources":srcs})

        else:
            # 수동 RAG 폴백: 문서 검색 + 대화맥락을 직접 프롬프트에 주입
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



# 랜덤 문제 생성기  채점
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

# 문제 표시 답안 입력  채점
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
        # 채점

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
