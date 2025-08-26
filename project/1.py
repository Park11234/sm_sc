import os
import tempfile
from typing import List
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import textwrap
import re

# =============================================
# File: pages/1_포토리소그래피.py
# =============================================
import streamlit as st
from langchain.chains import RetrievalQA

st.set_page_config(page_title="포토리소그래피", layout="wide")

st.header("1) 포토리소그래피")

# 개요
st.subheader("개요")
st.write("웨이퍼 표면에 감광막을 바르고 노광·현상으로 패턴을 형성합니다.")

# 핵심 포인트 (툴팁 기능 포함)
st.subheader("핵심 포인트")
st.markdown("""
- <span title="감광막을 웨이퍼에 균일하게 도포하는 단계입니다.">PR 코팅</span> → 
  <span title="감광막의 용매를 증발시켜 안정화시키는 단계입니다.">소프트베이크</span> → 
  <span title="마스크 패턴을 빛(EUV/DUV)을 통해 감광막에 전사하는 단계입니다.">노광(EUV/DUV)</span> → 
  <span title="노광 후 Bake를 통해 화학 반응을 안정화시키는 단계입니다.">PEB</span> → 
  <span title="노광된 영역을 현상액으로 제거하여 패턴을 형성하는 단계입니다.">현상</span> → 
  <span title="패턴을 고정하고 내열성 및 내화학성을 강화하는 단계입니다.">하드베이크</span> → 
  <span title="패턴의 결함 여부, 정렬 상태 등을 검사하는 단계입니다.">검사</span>
""", unsafe_allow_html=True)

st.markdown("- 해상도(λ, NA, k1), 포커스/도즈, LER/LWR")

st.subheader("프로세스 다이어그램")
steps = ["Wafer Clean", "PR Coat", "Soft Bake", "Exposure", "PEB", "Develop", "Hard Bake", "Inspection"]
st.graphviz_chart("\n".join([
    "digraph G {",
    "rankdir=LR;",
    "node [shape=box, style=rounded, fontsize=12];",
    *[f"n{i} [label=\"{s}\"];" for i, s in enumerate(steps)],
    *[f"n{i} -> n{i + 1};" for i in range(len(steps) - 1)],
    "}",
]), use_container_width=True)

st.subheader("공정 단계 설명")
steps_data = [
    {"name": "웨이퍼 세정 (Wafer Clean)", "desc": "웨이퍼 표면의 오염물, 입자 등을 제거하여 다음 공정의 품질을 확보합니다.", "icon": "🧼"},
    {"name": "감광막 도포 (PR Coat)", "desc": "감광막을 웨이퍼에 균일하게 도포합니다.", "icon": "🧴"},
    {"name": "소프트 베이크 (Soft Bake)", "desc": "감광막의 용매를 증발시켜 안정화시키고, 노광 시 패턴 품질을 높입니다.", "icon": "🔥"},
    {"name": "노광 (Exposure)", "desc": "마스크 패턴을 빛(EUV/DUV)을 통해 감광막에 전사합니다.", "icon": "💡"},
    {"name": "PEB (Post-Exposure Bake)", "desc": "노광 후 Bake를 통해 화학 반응을 안정화시키고 해상도를 향상시킵니다.", "icon": "♨️"},
    {"name": "현상 (Develop)", "desc": "노광된 영역을 현상액으로 제거하여 패턴을 형성합니다.", "icon": "🧪"},
    {"name": "하드 베이크 (Hard Bake)", "desc": "패턴을 고정하고 내열성 및 내화학성을 강화합니다.", "icon": "🧱"},
    {"name": "검사 (Inspection)", "desc": "패턴의 결함 여부, 정렬 상태 등을 검사하여 품질을 확인합니다.", "icon": "🔍"}
]

for step in steps_data:
    with st.expander(f"{step['icon']} {step['name']}"):
        st.write(step["desc"])

st.subheader("질의응답 (RAG)")
if "vectorstore" not in st.session_state:
    st.info("임베딩 자료가 없습니다. 메인에서 PDF 업로드 → 임베딩 생성 후 이용하세요.")
else:
    if "qa_chain" not in st.session_state:
        backend = st.session_state.get("llm_backend", "openai")
        model = st.session_state.get("llm_model", "gpt-4o-mini")
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        if backend == "openai":
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=model, temperature=0.2)
        else:
            from langchain_community.chat_models import ChatOllama

            llm = ChatOllama(model=model, temperature=0.2)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                                                return_source_documents=True)

    q = st.text_input("질문을 입력하세요", placeholder="예: EUV와 DUV 차이")
    if st.button("질문하기", use_container_width=True):
        if q.strip():
            out = st.session_state.qa_chain({"query": q})
            st.markdown("### 답변")
            st.write(out.get("result", "정보가 부족합니다"))
            src = out.get("source_documents") or []
            if src:
                with st.expander("출처"):
                    for i, s in enumerate(src, 1):
                        meta = s.metadata or {}
                        st.write(f"{i}. {meta.get('source', '파일')} p.{meta.get('page', '?')}")
        else:
            st.warning("질문을 입력하세요.")
# ================== 랜덤 문제 생성기 + 답안 채점 (카테고리별) ==================
import re, textwrap

st.subheader("랜덤 문제 생성기")

# 1) 카테고리명만 페이지별로 바꾸세요
CATEGORY_NAME = "포토리소그래피"  # ← 페이지 주제에 맞게

# 2) 난이도/문항수/컨텍스트 설정
cols = st.columns(3)
difficulty   = cols[0].selectbox("난이도", ["초급", "중급", "고급"], index=0)
n_items      = cols[1].selectbox("문항 수", [1, 3, 5], index=1)
has_vs       = "vectorstore" in st.session_state
with_context = cols[2].checkbox("업로드 문서 기반(권장)", has_vs)

# 3) LLM 준비 (OpenAI / Gemini, 폴백 가능)
def _get_llm_backend():
    return st.session_state.get("llm_backend", "openai"), st.session_state.get("llm_model", "gpt-4o")

def _generate_with_openai(prompt: str, model_name: str) -> str:
    # langchain-openai 있으면 사용, 없으면 OpenAI SDK 폴백
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model_name, temperature=0)
        return llm.invoke(prompt).content
    except Exception:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"__GEN_ERROR__ {e}"

def _generate_with_gemini(prompt: str, model_name: str) -> str:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        return llm.invoke(prompt).content
    except Exception:
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return resp.text
        except Exception as e:
            return f"__GEN_ERROR__ {e}"

def _gather_context(k: int = 6) -> str:
    if not has_vs or not with_context:
        return ""
    try:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(f"{CATEGORY_NAME} 핵심 개념 요약")
        return "\n\n".join(d.page_content for d in docs)[:6000]
    except Exception:
        return ""

def _extract_questions(s: str, expected_n: int) -> list[str]:
    "번호/불릿/빈줄 기준으로 최대 expected_n개 문제를 깔끔히 분리"
    s = s.strip()
    if not s:
        return []
    # 1) 번호 패턴으로 분리
    parts = re.split(r'^\s*\d+[\.\)\]]\s+', s, flags=re.M)
    parts = [p.strip() for p in parts if p.strip()]
    # 번호가 없으면 2) 빈 줄 기준
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r'\n\s*\n+', s) if p.strip()]
    # 그래도 하나뿐이면 3) 줄 단위로 자르기(문장이 길면 그대로 1문항)
    if len(parts) <= 1:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        parts = [(" ".join(lines))] if lines else []
    return parts[:expected_n]

QUIZ_PROMPT_TMPL = """\
당신은 반도체 공정 과목의 교수입니다.
주제: {category}
난이도: {difficulty}
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

판정을 한 후 피드백을 하세요 반드시 아래 형식을 정확히 지키세요 (줄바꿈 포함, 다른 텍스트 금지):
판정: 정답|오답
피드백: <두세 문장 피드백>
"""


# --- 4) 문제 생성 ---
if st.button("랜덤 문제 생성", use_container_width=True):
    backend, model = _get_llm_backend()
    context = _gather_context()
    prompt = QUIZ_PROMPT_TMPL.format(
        category=CATEGORY_NAME, difficulty=difficulty, n_items=n_items,
        context=(f"[컨텍스트]\n{context}" if context else "(컨텍스트 없음)")
    )
    quiz_text = _generate_with_openai(prompt, model) if backend == "openai" else _generate_with_gemini(prompt, model)

    # 파싱해서 리스트로 저장
    items = _extract_questions(quiz_text, n_items)
    st.session_state[f"{CATEGORY_NAME}_quiz_items"] = items
    # 답안 초기화
    for i in range(len(items)):
        st.session_state.pop(f"{CATEGORY_NAME}_ans_{i}", None)

# --- 5) 문제 표시 + 답안 입력 ---
items = st.session_state.get(f"{CATEGORY_NAME}_quiz_items", [])
if items:
    st.markdown("### 생성된 문제")
    for i, qtext in enumerate(items, start=1):
        st.markdown(f"**{i}) {qtext}**")
        st.text_area(
            f"답안 입력 #{i}",
            key=f"{CATEGORY_NAME}_ans_{i-1}",
            height=100,
            placeholder="여기에 본인 답안을 작성하세요."
        )
    def _format_eval(judged: str) -> str:
        """LLM 출력에서 '판정'과 '피드백'을 분리해 항상 줄바꿈 형태로 반환"""
        s = judged.strip().replace('\r\n', '\n')

        # 1) 판정/피드백 추출
        m_verdict = re.search(r'판정\s*:\s*(정답|오답)', s)
        m_feedback = re.search(r'피드백\s*:\s*(.*)', s, flags=re.S)

        if m_verdict:
            verdict = m_verdict.group(1).strip()
            feedback = (m_feedback.group(1).strip() if m_feedback else "")
            # 2) 최종 포맷 (항상 줄바꿈)
            return f"판정: {verdict}\n피드백: {feedback}" if feedback else f"판정: {verdict}\n피드백: (없음)"

        # 3) 판정 태그가 없으면, '피드백:' 앞에 강제 줄바꿈 삽입
        s = re.sub(r'\s*(피드백\s*:)', r'\n\1', s).strip()
        return s

    # --- 6) 채점 ---
    if st.button("채점하기", type="primary", use_container_width=True):
        backend, model = _get_llm_backend()
        context = _gather_context()
        results = []
        for i, qtext in enumerate(items):
            ans = st.session_state.get(f"{CATEGORY_NAME}_ans_{i}", "").strip()
            eval_prompt = EVAL_PROMPT_TMPL.format(
                category=CATEGORY_NAME,
                question=qtext,
                answer=ans if ans else "(무응답)",
                context=(f"[컨텍스트]\n{context}" if context else "(컨텍스트 없음)")
            )
            judged = _generate_with_openai(eval_prompt, model) if backend == "openai" else _generate_with_gemini(eval_prompt, model)
            results.append(judged)

        st.markdown("### 채점 결과")
        for i, judged in enumerate(results, start=1):
            st.markdown(f"**문항 #{i} 결과**")
            st.markdown(_format_eval(judged))

else:
    st.caption("아직 생성된 문제가 없습니다. ‘랜덤 문제 생성’을 눌러주세요.")
