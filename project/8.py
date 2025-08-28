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

st.set_page_config(page_title="CMP", layout="wide")

st.header("8) 평탄화 (CMP)")

CATEGORY_NAME = "평탄화"

st.subheader("개요")
st.write("연마 패드와 슬러리를 사용해 표면을 평탄화하는 공정입니다.")

st.subheader("핵심 포인트")
st.markdown("""
- 패드/슬러리/다운포스/상대속도 제어
- 디싱·스크래치·오버폴리시 제어, 엔드포인트 관리
""")

st.subheader("프로세스(가로 스크롤 카드)")
steps = ["패드 컨디셔닝", "연마 (슬러리 사용)", "종점 제어", "세정", "계측 (메트롤로지)"]

html = """
<div style="overflow-x:auto; padding:6px 0;">
  <div style="display:flex; gap:12px; align-items:center; min-height:96px;">
    {items}
  </div>
</div>
"""
chip = """
<div style="flex:0 0 auto; min-width:160px; max-width:220px;
            padding:12px 14px; border:1.5px solid #d0d4dc; border-radius:14px;
            box-shadow:0 1px 3px rgba(0,0,0,0.06); background:#fff; text-align:center;">
  <div style="font-size:14px; font-weight:600;">{label}</div>
</div>
<div style="flex:0 0 auto; font-size:20px; margin:0 2px;">➜</div>
"""

items = "".join(
    (chip.format(label=s) if i < len(steps)-1
     else chip.replace('<div style="flex:0 0 auto; font-size:20px; margin:0 2px;">➜</div>', '').format(label=s))
    for i, s in enumerate(steps)
)

import streamlit.components.v1 as components
components.html(html.format(items=items), height=120, scrolling=False)

# 공정 단계 설명

st.subheader("공정 단계 설명 및 진도 관리")

# 단계별 정보
steps_data = [
    {
        "id": "cmp_pad_cond",
        "name": "패드 컨디셔닝 (Pad Conditioning)",
        "icon": "🧽",
        "desc": """
🧽 **설명: 패드 컨디셔닝 (Pad Conditioning)**

CMP 공정의 시작 단계로, 패드 표면을 다이아몬드 컨디셔너로 긁어내어
패드의 기공을 열어주고 연마 효율(Removal Rate, RR)을 안정화시킵니다.

🔧 **주요 포인트**
- 컨디셔닝 강도/시간이 RR에 큰 영향을 미침
- 과도하면 스크래치 증가, 부족하면 글레이징(막힘) 발생
- 슬러리 유입 경로 확보 및 표면 활성화가 목적

🎯 핵심 요약:
- Pad Conditioning은 CMP 공정 품질의 출발점
- RR 안정화와 결함 방지를 동시에 달성해야 함
"""
    },
    {
        "id": "cmp_polish",
        "name": "연마 (Polish - Slurry 사용)",
        "icon": "🌀",
        "desc": """
🧴 **설명: 연마 (Polish with Slurry)**

슬러리(Slurry)는 연마제(입자) + 화학 반응제가 섞인 용액으로,
패드와 함께 웨이퍼 표면을 기계적·화학적으로 동시에 깎아냅니다.

🔧 **주요 제어 인자**
- 패드 종류 (경도, 기공률, 탄성)
- 슬러리 조성 (pH, 산화제, 억제제)
- 다운포스 (압력, psi 단위)
- 상대속도 (Pad/Wafer 회전 속도)

⚠️ **주의사항**
- 다운포스 ↑ → RR ↑ but 디싱 위험
- 속도 ↑ → 균일성 개선 but 스크래치 위험
- 슬러리 농도/유량 안정화 필요

🎯 핵심 요약:
- CMP 제거률(RR)은 Pad/Slurry/압력/속도의 균형으로 제어
"""
    },
    {
        "id": "cmp_ep",
        "name": "엔드포인트 제어 (Endpoint Control)",
        "icon": "🎯",
        "desc": """
🔍 **설명: 엔드포인트 제어 (Endpoint Control)**

연마 중 목표 두께 또는 특정 층이 드러났을 때, 
과도 연마(Over-Polish)를 막고 균일성을 확보하기 위한 제어 단계입니다.

📏 **측정 방식**
- 광학 센서: 반사율 변화 감지
- 전기적 센서: 금속 층 노출 시 전류/저항 변화
- 토크/마찰 센서: 패드-웨이퍼 마찰력 변화 측정
- 시간 기반 + 센서 결합: 안정적 엔드포인트 검출

⚠️ **문제 예방**
- 오버폴리시 → 라인 CD 손상, 두께 불량
- 언더폴리시 → 절연/배선 잔류

🎯 핵심 요약:
- EP 제어는 수율을 좌우하는 안전장치
- 센서+시간 융합 제어로 재현성 확보
"""
    },
    {
        "id": "cmp_clean",
        "name": "세정 (Clean)",
        "icon": "🧼",
        "desc": """
🧼 **설명: 세정 (Clean)**

CMP 후 웨이퍼 표면에는 슬러리 입자, 화학 잔여물, 금속 이온 등이 남아있습니다.  
이를 완벽히 제거하지 않으면 후속 공정에서 **결함, 오염, 신뢰성 문제**를 일으킵니다.

🔧 **세정 방식**
- DIW(초순수) 세정
- 브러시 스크럽
- 메가소닉 클리닝
- 화학 세정 (산/염기)

⚠️ **주의사항**
- 과세정 → 표면 손상
- 불충분 세정 → 파티클 잔류

🎯 핵심 요약:
- CMP 세정은 **오염 제거 = 수율 보증**
- 클린룸 관리와 병행 필요
"""
    },
    {
        "id": "cmp_metrology",
        "name": "계측 (Metrology)",
        "icon": "📐",
        "desc": """
📏 **설명: 계측 (Metrology)**

CMP 후 최종적으로 표면 상태와 두께, 평탄도를 측정하는 단계입니다.  
이 결과는 다음 공정 진행 여부와 피드백 제어에 활용됩니다.

🔧 **측정 항목**
- 두께/평탄도: 엘립소미터, 옵틱 두께계
- 결함: 광학 검사, SEM, AFM
- 금속 배선: 저항 측정, 라인 CD 확인

🎯 핵심 요약:
- 계측은 CMP 품질 검증의 마지막 단계
- 데이터는 공정 조건 보정에 필수
"""
    }
]

# 페이지 진도 버킷
ids = [s["id"] for s in steps_data]
if "progress" not in st.session_state:
    st.session_state.progress = {}
for sid in ids:
    st.session_state.progress.setdefault(sid, False)

# 단계별 설명 및 체크박스
completed = 0
for step in steps_data:
    sid, name, icon = step["id"], step["name"], step["icon"]
    with st.expander(f"{icon} {name}"):
        st.write(step["desc"])
        widget_key = f"done_{sid}"
        checked = st.checkbox("이 단계 학습 완료",
                              value=st.session_state.progress.get(sid, False),
                              key=widget_key)
        st.session_state.progress[sid] = checked
        if checked:
            completed += 1

# 전체 진도율 표시
total = len(steps_data)
percent = int((completed / total) * 100) if total else 0
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


# 체인 준비
if "vectorstore" not in st.session_state:
    st.info("임베딩 자료가 없습니다. 메인에서 PDF 업로드 → 임베딩 생성 후 이용하세요.")
else:
    if "qa_chain" not in st.session_state:
        # LLM.py의 함수로 백엔드/모델/LLM을 가져옴.
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

    # 과거 채팅 표시 (최근 2개만 보이고 나머지는 펼쳐서 볼 수 있음)
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

# (중복 회피용)
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
        채점
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

