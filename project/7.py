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

st.set_page_config(page_title="금속배선", layout="wide")

st.header("7) 금속배선 (Metallization)")

CATEGORY_NAME = "금속배선"

st.subheader("개요")
st.write(
    "소자 간 전기적 연결을 형성하는 공정으로, 다마신(damascene) 구조를 기반으로 "
    "포토/식각 → 배리어/라이너 → 시드 → 금속 충전(도금/증착) → CMP → 캡/패시베이션 순으로 진행됩니다. "
    "저저항(ρ), 낮은 접촉저항(Rc), 신뢰성(EM/SM/TDDB), 공극/시임 무결점, 저유전막(Low-k) 호환성이 핵심입니다."
)

# 핵심 포인트 (툴팁)
st.subheader("핵심 포인트")
st.markdown("""
- <span title="Low-k/하드마스크 패터닝, 트렌치·비아 정의">포토/식각</span> →
  <span title="확산 방지·접착력 향상(예: Ta/TaN, Ru, Co)">배리어/라이너</span> →
  <span title="균일 충전을 위한 얇고 연속적인 Cu 시드 형성(PVD/ALD)">시드</span> →
  <span title="ECP 첨가제(억제제/가속제/조절제)로 공극 없는 충전">금속 충전</span> →
  <span title="과충전 제거·평탄화, 다마신 구현">CMP</span> →
  <span title="어닐/캡층(SiCN/SiN/Ru)으로 저항/EM 신뢰성 향상">후처리/캡</span> →
  <span title="비아저항/라인저항/EM·TDDB/결함 맵핑">검사/계측</span>
""", unsafe_allow_html=True)
st.markdown("- 핵심 지표: 라인/비아 저항, 접촉저항 Rc, 공극/시임 결함, 배리어 연속성, 시드 커버리지, EM/SM 수명, 저유전막 손상, 평탄도")

# 프로세스(스크롤 카드)
st.subheader("프로세스")
steps = [
    "포토/식각(Trench/Via)",
    "배리어/라이너 증착",
    "시드 증착(Seed)",
    "금속 충전(Fill)",
    "CMP 평탄화",
    "후처리/캡",
    "검사/계측"
]

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
        "name": "포토/식각(Trench/Via)",
        "icon": "🧩",
        "desc": """
🧩 **포토/식각(Trench/Via)**
- **식각 대상**: Low-k 유전 막에 트렌치/비아를 형성해 라인·비아 경로를 만듭니다(싱글/더블 다마신).
- **방법**: 하드마스크( SiCN/SiN ) 사용, 플라즈마 식각(예: Fluorocarbon)으로 ARDE 최소화, PR 스트립/플라즈마 폴리머 제거.
- **주의사항**
  - Low-k 플라즈마 손상/수분 흡수 → 유전율 상승·누설↑, **저손상 식각·세정** 필요.
  - 비아 바닥 **Stop 레이어** 노출 과소/과다 시 접촉저항 변동.
  - PR/폴리머 잔류는 금속 충전 시 **공극/시임** 유발.
- 🎯 **핵심 요약**: 저손상 식각과 잔류 제거가 이후 **충전성·Rc**의 기반.
"""
    },
    {
        "name": "배리어/라이너 증착",
        "icon": "🧱",
        "desc": """
🧱 **배리어/라이너 증착(Barrier/Liner)**
- **역할**: 금속(Cu 등)의 **확산 방지**, Low-k/하부막과 **접착력** 향상, 전기·기계 신뢰성 확보.
- **재료/방법**
  - Cu 다마신: **Ta/TaN**(배리어) + **Ru/Co**(라이너 대안) / PVD, CVD, ALD.
  - W 콘택/비아: **Ti/TiN** 배리어 + **W CVD** 필.
- **파라미터**: 콘포멀리티, 두께(너무 두꺼우면 단면 축소→R↑), 연속성(핀홀 無), 응력/접착.
- **주의사항**
  - 고 AR 구조에서 PVD 배리어 **사이드월 커버리지 부족** → ALD/CVD 보완.
  - 배리어 과두께는 유효 단면 축소로 라인저항 상승.
  - **오염·수분** 존재 시 계면 박리/EM 열화.
- 🎯 **핵심 요약**: **연속·얇음·콘포멀** 3요건 충족이 핵심(특히 고 AR 비아).
"""
    },
    {
        "name": "시드 증착(Seed)",
        "icon": "🌱",
        "desc": """
🌱 **시드 증착(Seed)**
- **역할**: 전해 도금(ECP) 전류 흐름을 위한 연속 금속 경로 제공(주로 **Cu PVD/ALD**).
- **품질 기준**: **연속성**(no open), 두께 균일도, 고 AR 비아 **사이드월 커버리지**, 산화/오염 無.
- **보강**: ALD 시드/선두층, 경사 증착/바이어스 보조, 프리클린(H₂/Ar).
- **주의사항**
  - 시드 불연속/산화 → 도금 초기 **누드 영역** 발생 → 공극/시임.
  - 두께 과다 시 유효 단면 축소, 과소 시 저항↑·충전성↓.
- 🎯 **핵심 요약**: **끊김 없는 시드**가 공극 없는 ECP의 선결조건.
"""
    },
    {
        "name": "금속 충전(Fill)",
        "icon": "⚡",
        "desc": """
⚡ **금속 충전(Fill)**
- **Cu ECP**: 첨가제(억제제, 가속제, 조절제) 조합으로 **상향/전면 성장 균형** → 공극/시임 억제.
- **W CVD**: SiH₄/ WF₆ 기반 핵생성 후 벌크 성장(콘택/비아에 적합).
- **Co/Ru 필**(첨단 노드 대안): 우수한 라이너 호환/EM 내성.
- **파라미터**: 전류 밀도/파형(DC/펄스), 온도, 용액/전구체 청정도, 교반, 에이징.
- **주의사항**
  - ECP **과전류/첨가제 밸런스 붕괴** → 공극/시임, 표면 거칠기↑.
  - 용액/전구체 오염, 미세 기포/흐름 불균일 → 국부 결함.
- 🎯 **핵심 요약**: **첨가제 밸런스 + 전류 프로파일**로 고 AR 구조도 무결점 충전.
"""
    },
    {
        "name": "CMP 평탄화",
        "icon": "🧽",
        "desc": """
🧽 **CMP 평탄화(Chemical Mechanical Planarization)**
- **역할**: 과충전된 금속/배리어를 제거하고 **평탄도** 확보, 다마신 패턴 완성.
- **구성**: 슬러리(산화제/부식억제제/연마제), 패드, 다운포스/속도, 엔드포인트.
- **결함 관리**: 디싱/언더컷, 스크래치, 오염/갭핑, 배리어 선택비.
- **주의사항**
  - Low-k/유전막 손상·수분 흡수, 메탈 **Galvanic** 반응 주의.
  - 오버폴리시 → 라인 단면 축소·저항↑, 언더폴리시 → 잔류/쇼트.
- 🎯 **핵심 요약**: **선택적 제거 + 평탄도**를 지키면서 디싱/스크래치를 최소화.
"""
    },
    {
        "name": "후처리/캡",
        "icon": "🛡️",
        "desc": """
🛡️ **후처리/캡(Post / Cap Layer)**
- **어닐**: Cu 재결정/결정립 성장 → 저항↓, EM 내성↑. (RTA/퍼니스, N₂/H₂/Ar)
- **캡 층**: SiCN/SiN/Ru 등으로 Cu **확산 차단**·표면 보호, 차세대는 **셀프 폼드 캡(SFC)**도 사용.
- **표면 개질**: 플라즈마/UV-O₃로 오염 제거·접착 향상.
- **주의사항**
  - 과도한 열 예산은 Low-k 변성/수축, 라인 스트레스 변화.
  - 캡 불연속/핀홀은 EM/TDDB 취약점.
- 🎯 **핵심 요약**: **어닐 + 캡**으로 저항과 신뢰성을 동시에 끌어올림.
"""
    },
    {
        "name": "검사/계측",
        "icon": "🔍",
        "desc": """
🔍 **검사/계측(Metrology)**
- **전기**: 라인/비아 저항( Kelvin / 4PP ), 접촉저항(Rc), EM/SM 가속 테스트, TDDB.
- **구조/표면**: 크로스섹션 SEM/FIB-TEM(공극/시임/배리어 연속성), AFM(거칠기), XRR/XRD(밀도/응력/결정상).
- **화학**: ToF-SIMS/XPS(오염/확산), 잔류 첨가제/세정제 분석.
- **주의사항**
  - 측정 레시피 고정·레퍼런스 유지, Layout 의존성(패턴 밀도/피치) 고려.
  - 온습도/보관에 따른 Low-k 수분 흡수 관리.
- 🎯 **핵심 요약**: **전기 + 단면**을 함께 보며 결함 원인을 빠르게 역추적, 레시피에 피드백.
"""
    },
]

# 페이지 진도 버킷
PAGE_PROGRESS_KEY = f"{CATEGORY_NAME}_progress"
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
            key=f"{CATEGORY_NAME}_{s['name']}"   # ← 페이지 네임스페이스로 키 충돌 방지
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

        # (선택) 체인 재생성을 원하시면 아래 주석을 해제


        st.toast("질의응답 대화가 초기화되었습니다. 🧹")
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()


# PDF 자료 넣기
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
                # CRC 불가(비호환 LLM 등) → 수동 RAG로 처리
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
