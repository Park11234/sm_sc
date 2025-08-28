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

st.set_page_config(page_title="산화", layout="wide")

# 페이지 본문
st.header("3) 산화 (Oxidation)")

CATEGORY_NAME = "산화"

# 개요
st.subheader("개요")
st.write(
    "웨이퍼 표면의 Si를 산화시켜 SiO₂ 막을 형성하는 공정입니다. 게이트 산화막/필드 산화 등에서 핵심 역할을 하며, "
    "건식(드라이)·습식(웨트) 산화, 증발/확산 속도, 두께 균일도, 계면 상태(Qf, Dit) 관리가 중요합니다."
)

# 핵심 포인트
st.subheader("핵심 포인트")
st.markdown("""
- <span title="유기물/금속이온/파티클 제거 및 표면 수분 관리">전처리 세정</span> →
  <span title="건식(O₂) 또는 습식(H₂O) 산화 분위기 설정">분위기 설정</span> →
  <span title="Deal–Grove 모델 기반 시간/온도/분압 제어로 목표 두께 형성">열 산화</span> →
  <span title="냉각 및 잔류 스트레스 완화">어닐/쿨다운</span> →
  <span title="두께/굴절률/계면전하/누설 평가">검사/계측</span>
""", unsafe_allow_html=True)
st.markdown("- 핵심 지표: 두께(Uniformity), 굴절률, 계면 상태(Dit/Qf), 누설, 스트레스/필름 품질")

# 가로 스크롤 박스
st.subheader("프로세스(가로 스크롤 카드)")
steps = ["전처리 세정(Pre-clean)", "분위기 설정(Ambient)", "열 산화(Thermal Ox)",
         "어닐/쿨다운(Anneal/Cool)", "검사/계측(Metrology)"]

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

# 단계 정보
steps_data = [
    {
        "name": "전처리 세정 (Pre-clean)",
        "icon": "🧼",
        "desc": """
🧼 **전처리 세정(Pre-clean)**

**왜 세정이 중요한가?**
- 표면 유기물, 금속 이온, 파티클, 흡착 수분 제거 → 산화 초기 계면 품질 확보(낮은 Dit, 안정적 Qf).
- 네이티브 옥사이드 상태 통일 → 두께 재현성/균일도 향상.

**세정 방식**
- SC-1(NH₄OH:H₂O₂:H₂O)로 유기물/파티클 제거 → DIW 린스.
- SC-2(HCl:H₂O₂:H₂O)로 금속 이온 제거 → DIW 린스.
- HF-last(희박 HF)로 얇은 네이티브 산화막 제거 후 **신속 로딩**(재산화 방지).
- 드라이(스핀/IPA 증기/베이크)로 수분 제거.

**주의사항**
- DIW 저항(메가옴) 관리, 린스/드라이 불충분 시 워터마크·파티클 유발.
- HF 과공정 → 표면 거칠기/마이크로피트 발생 가능.
- HF-last 이후 대기 노출 시간 최소화(재산화·오염).

🎯 **핵심 요약**
- ‘HF-last + 빠른 로딩’이 얇은 산화막 품질의 출발점.
"""
    },
    {
        "name": "분위기 설정 (Ox Ambient)",
        "icon": "🌫️",
        "desc": """
🌫️ **분위기 설정(Ambient)**

**선택지**
- **건식(Dry O₂)**: 반응 속도 낮지만 막질/전기적 특성 우수(게이트 산화막).
- **습식(Steam, H₂O)**: 속도 빠르고 두꺼운 산화막 용이(필드 산화).

**조건/장비**
- 퍼니스(수평/수직) 또는 RTO(Rapid Thermal Oxidation).
- 분압/유량/온도 램프·소크 제어, 퍼지( N₂ )로 잔류 가스 제거.
- 스팀: 버블러/피로제닉( H₂ + O₂ → H₂O ) 방식 선택.

**주의사항**
- 튜브 메모리 효과(이전 공정 레시피 잔류) → 프리컨디셔닝/더미 웨이퍼 사용.
- 하드웨어 클린 주기, 가스 라인 수분/오일 관리.
- 로딩 순서·캐리어 위치 의존성 → 균일도 영향.

🎯 **핵심 요약**
- 목적(막질 vs 속도)에 맞춰 Dry/Wet 선택, **분위기 안정화**가 재현성의 핵심.
"""
    },
    {
        "name": "열 산화 (Thermal Ox)",
        "icon": "🔥",
        "desc": """
🔥 **열 산화(Thermal Oxidation)**

**메커니즘/모델**
- Deal–Grove: 초기 **선형 지배**(계면 반응) → 두꺼워질수록 **확산 지배**(산화종 확산).
- 두께 \(x\)는 \(B/A\)·\(B\)(선형/포물 상수)와 **시간·온도·분압**의 함수.

**주요 파라미터**
- 온도: 800~1100 ℃ (고온일수록 속도↑, 결함/스트레스 주의).
- 시간: 타깃 두께에 맞춘 계산/캘리브레이션(레이디얼 보정 포함).
- 도핑/결정방향: 산화 속도/막질에 영향(예: p형, (111) 등).
- Dry vs Wet: Dry는 얇고 고품질, Wet은 두껍고 빠름.

**프로파일/품질**
- 두께 균일도(웨이퍼 내/웨이퍼 간), 막질(밀도/결합), 계면 거칠기 최소화.
- NO/N₂O 얇은 질화(산질화막)로 누설/신뢰성 개선 옵션.

**주의사항**
- 과도한 열 예산 → 금속 확산/디바이스 열 손상.
- 스택 상호작용(예: 상부막 유무)로 산화 속도 변화.
- 장시간 Wet 산화 시 스트레스·워프·슬립 라인 관리.

🎯 **핵심 요약**
- **모델 기반 레시피(시간·온도·분압) + 장비 캘리브레이션**으로 목표 두께·막질 달성.
"""
    },
    {
        "name": "어닐/쿨다운 (Anneal/Cool)",
        "icon": "🧊",
        "desc": """
🧊 **어닐/쿨다운**

**목적**
- 계면 결함 패시베이션( Dit 저감), 전하 안정화(Qf), 스트레스 릴리프.

**방법**
- PDA(Post-Deposition Anneal) in N₂/H₂(포밍가스, 5~10% H₂), RTA/퍼니스 선택.
- 램프 업/다운 레이트 제어로 슬립/워프/크랙 억제.
- 필요 시 저온 플라즈마 처리로 표면 결함 보정.

**주의사항**
- H₂ 취급 안전, 메탈 존재 시 어닐 온도 윈도우 준수.
- 급냉은 미세균열/워프 유발, 과도한 장시간 어닐은 도핑 프로파일 변형.

🎯 **핵심 요약**
- **포밍가스 PDA**로 계면 품질 향상, **완만한 램프/쿨다운**으로 기계적 결함 예방.
"""
    },
    {
        "name": "검사/계측 (Metrology)",
        "icon": "🔍",
        "desc": """
🔍 **검사/계측**

**두께/굴절률**
- 타원계/반사율로 두께/ n,k 측정(열 SiO₂ 기준 n≈1.46 @632.8 nm).
- 맵 측정으로 레디얼 균일도/로트 간 변동 분석.

**전기/계면**
- C–V(금속–산화막–반도체, MOSCAP)로 Dit/Qf, 브레이크다운/누설 전류 평가.
- I–V, TDDB로 신뢰성 확인.

**화학/구조**
- FTIR(결합 상태), XPS(조성/결합 에너지), TEM/XTEM(계면/두께 단면 관찰).

**주의사항**
- 측정 파라미터(각도/파장/모델) 불일치 시 오차 증가 → 레시피 고정/레퍼런스 사용.
- 고도핑/박막 영역은 보정 필요.

🎯 **핵심 요약**
- **두께·균일도 + C–V로 계면 품질**을 정량화하고 결과를 공정에 피드백.
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
            key=f"{CATEGORY_NAME}_{s['name']}"
        )
        st.session_state[PAGE_PROGRESS_KEY][s["name"]] = checked
        if checked: completed += 1

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
        # LLM.py의 함수로 백엔드/모델/LLM을 가져옴.
        # "openai" | "gemini", 모델 문자열
        try:
            backend, model = get_llm_backend()
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
        # 기본 수동이지만  CRC 되면 "crc"로 변경
        st.session_state.qa_mode = "manual"

        if llm is not None:
            try:
                # CRC 시도 (이전 대화 맥락을 직접 넘길 수 있음)
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

    # ✅ 버튼을 눌렀고 비어 있지 않을 때만 생성
    if submitted and user_q and user_q.strip():
        # 1) 사용자 메시지 기록
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
            #수동 RAG 폴백: 문서 검색 + 대화맥락/발췌를 직접 프롬프트에 주입
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

# 랜덤 문제 생성기 + 채점
st.subheader("랜덤 문제 생성기")
CATEGORY_NAME = "포토리소그래피"  # ← 페이지 주제명

# (중복 회피용)
hist_key = f"{CATEGORY_NAME}_quiz_history"
if hist_key not in st.session_state:
    # 문자열(서술형 질문) 또는 MC 질문 텍스트 저장
    st.session_state[hist_key] = []

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

# 프롬프트 템플릿
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

# 문제 표시 + 답안 입력 / 채점
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
