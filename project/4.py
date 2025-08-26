import os
import tempfile
from typing import List
import streamlit as st

# LangChain / Vector DB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# =============================================
# File: pages/4_확산.py
# =============================================
import streamlit as st
from langchain.chains import RetrievalQA
import re

st.set_page_config(page_title="확산", layout="wide")

st.header("4) 확산 (Diffusion)")

st.subheader("개요")
st.write("고온에서 도펀트를 확산시켜 농도 프로파일을 형성합니다.")

st.subheader("핵심 포인트")
st.markdown("""
- Pre-Deposition → Drive-in
- 가우시안/에러펑션 프로파일, xj/저항 타깃팅
""")

st.subheader("프로세스 다이어그램")
steps = ["Pre-Deposition", "Drive-In", "Anneal", "Metrology"]
st.graphviz_chart("\n".join([
    "digraph G {",
    "rankdir=LR;",
    "node [shape=box, style=rounded, fontsize=12];",
    *[f"n{i} [label=\"{s}\"];" for i, s in enumerate(steps)],
    *[f"n{i} -> n{i + 1};" for i in range(len(steps) - 1)],
    "}",
]), use_container_width=True)

st.subheader("질의응답 (RAG)")
if "vectorstore" not in st.session_state:
    st.info("메인에서 임베딩을 먼저 생성하세요.")
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

    q = st.text_input("질문을 입력하세요", placeholder="예: Drive-in 목적")
    if st.button("질문하기", use_container_width=True):
        if q.strip():
            out = st.session_state.qa_chain({"query": q})
            st.markdown("### 답변")
            st.write(out.get("result", "정보가 부족합니다"))
            for i, s in enumerate(out.get("source_documents", []), 1):
                meta = s.metadata or {}
                st.caption(f"{i}. {meta.get('source', '파일')} p.{meta.get('page', '?')}")
        else:
            st.warning("질문을 입력하세요.")
# ---------------- 랜덤 문제 생성기 + 채점 ----------------
st.subheader("랜덤 문제 생성기")
CATEGORY_NAME = "확산"  # ← 페이지 주제명

# (중복 회피용 히스토리)
hist_key = f"{CATEGORY_NAME}_quiz_history"
if hist_key not in st.session_state:
    st.session_state[hist_key] = []  # 문자열(서술형 질문) 또는 MC 질문 텍스트 저장

# 설정
cols = st.columns(3)
# ✅ 난이도 두 가지만
difficulty   = cols[0].selectbox(
    "난이도",
    ["초급", "고급"],
    index=0,
    key=f"{CATEGORY_NAME}_difficulty"      # ← 고유 key 추가
)
n_items      = cols[1].selectbox(
    "문항 수",
    [1, 3, 5],
    index=1,
    key=f"{CATEGORY_NAME}_n_items"         # ← 고유 key 추가
)
has_vs       = "vectorstore" in st.session_state
with_context = cols[2].checkbox(
    "업로드 문서 기반(권장)",
    has_vs,
    key=f"{CATEGORY_NAME}_with_context"    # ← 고유 key 추가(권장)
)

# LLM 헬퍼 (기존과 동일)
def _get_llm_backend():
    return st.session_state.get("llm_backend", "openai"), st.session_state.get("llm_model", "gpt-4o")

def _generate_with_openai(prompt: str, model_name: str) -> str:
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

# ----- 공통: 서술형 추출기 (고급용)
def _extract_questions(s: str, expected_n: int) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = re.split(r'^\s*\d+[\.\)\]]\s+', s, flags=re.M)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r'\n\s*\n+', s) if p.strip()]
    if len(parts) <= 1:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        parts = [(" ".join(lines))] if lines else []
    return parts[:expected_n]

# ----- 초급(MC) 전용: 생성 템플릿 & 파서
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

def _parse_mc_questions(s: str, expected_n: int):
    """LLM 출력에서 객관식 문항을 구조화하여 [ {q:str, opts:list[A..D], answer:'A'..'D', expl:str}, ... ] 로 반환"""
    blocks = re.split(r'^\s*\d+\)\s+', s.strip(), flags=re.M)
    items = []
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        # 질문: 보기 시작 전 줄들 합치기
        q_lines = []
        opts = {'A': None, 'B': None, 'C': None, 'D': None}
        ans = None
        expl = ""
        phase = "q"
        for ln in lines:
            m_opt = re.match(r'^([ABCD])[)\.]\s*(.+)$', ln, flags=re.I)
            if m_opt:
                phase = "opt"
                key = m_opt.group(1).upper()
                opts[key] = m_opt.group(2).strip()
                continue
            m_ans = re.match(r'^정답\s*:\s*([ABCD])\s*$', ln, flags=re.I)
            if m_ans:
                ans = m_ans.group(1).upper()
                phase = "ans"
                continue
            m_ex = re.match(r'^해설\s*:\s*(.*)$', ln, flags=re.I)
            if m_ex:
                expl = m_ex.group(1).strip()
                phase = "expl"
                continue
            if phase == "q":
                q_lines.append(ln)
            elif phase == "expl":
                expl += (" " + ln)
            # 나머지는 무시
        qtext = " ".join(q_lines).strip()
        if qtext and all(opts[k] for k in ['A','B','C','D']) and ans in 'ABCD':
            items.append({
                "q": qtext,
                "opts": [f"A) {opts['A']}", f"B) {opts['B']}", f"C) {opts['C']}", f"D) {opts['D']}"],
                "answer": ans,
                "expl": expl.strip()
            })
        if len(items) >= expected_n:
            break
    return items

# ----- 고급(서술형) 템플릿
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

# ----- 채점 템플릿(서술형용, 기존 유지)
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

# ----- 문제 생성 -----
if st.button("랜덤 문제 생성", use_container_width=True):
    # 진행표시가 사라지도록 플레이스홀더
    ph = st.empty()
    with ph.container():
        if hasattr(st, "status"):
            with st.status("문제 생성 중...", expanded=True) as status:
                status.update(label="컨텍스트 수집...", state="running")
                backend, model = _get_llm_backend()
                context = _gather_context()

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
                raw = _generate_with_openai(prompt, model) if backend == "openai" else _generate_with_gemini(prompt, model)

                prev_texts = [p if isinstance(p, str) else p.get("q","") for p in st.session_state[hist_key]]

                if difficulty == "초급":
                    cand = _parse_mc_questions(raw, n_items)
                    uniques = []
                    for item in cand:
                        if not any(is_similar(item["q"], pt) for pt in prev_texts):
                            uniques.append(item)
                    # 부족 시 보강 1회
                    if len(uniques) < n_items:
                        need = n_items - len(uniques)
                        status.update(label=f"보강 생성 ({need}개)...", state="running")
                        raw2 = _generate_with_openai(prompt, model) if backend == "openai" else _generate_with_gemini(prompt, model)
                        cand2 = _parse_mc_questions(raw2, need)
                        for it in cand2:
                            if len(uniques) >= n_items: break
                            if not any(is_similar(it["q"], pt) for pt in (prev_texts + [u["q"] for u in uniques])):
                                uniques.append(it)

                    st.session_state[f"{CATEGORY_NAME}_quiz_items"] = uniques
                    st.session_state[f"{CATEGORY_NAME}_quiz_mode"]  = "초급"
                    st.session_state[hist_key].extend([u["q"] for u in uniques])

                else:  # 고급(서술형)
                    cand = _extract_questions(raw, n_items)
                    uniques = []
                    for q in cand:
                        if not any(is_similar(q, pt) for pt in prev_texts):
                            uniques.append(q)
                    if len(uniques) < n_items:
                        need = n_items - len(uniques)
                        status.update(label=f"보강 생성 ({need}개)...", state="running")
                        raw2 = _generate_with_openai(prompt, model) if backend == "openai" else _generate_with_gemini(prompt, model)
                        cand2 = _extract_questions(raw2, need)
                        for q in cand2:
                            if len(uniques) >= n_items: break
                            if not any(is_similar(q, pt) for pt in (prev_texts + uniques)):
                                uniques.append(q)

                    st.session_state[f"{CATEGORY_NAME}_quiz_items"] = uniques
                    st.session_state[f"{CATEGORY_NAME}_quiz_mode"]  = "고급"
                    st.session_state[hist_key].extend(uniques)

                status.update(label="완료 ✅", state="complete")
        else:
            # st.status 없는 구버전 호환: 간단 진행바
            bar = st.progress(0)
            backend, model = _get_llm_backend()
            context = _gather_context(); bar.progress(20)
            if difficulty == "초급":
                prompt = QUIZ_PROMPT_MC.format(category=CATEGORY_NAME, n_items=n_items, context=(f"[컨텍스트]\n{context}" if context else "(컨텍스트 없음)"))
            else:
                prompt = QUIZ_PROMPT_TXT.format(category=CATEGORY_NAME, n_items=n_items, context=(f"[컨텍스트]\n{context}" if context else "(컨텍스트 없음)"))
            raw = _generate_with_openai(prompt, model) if backend == "openai" else _generate_with_gemini(prompt, model); bar.progress(60)
            prev_texts = [p if isinstance(p, str) else p.get("q","") for p in st.session_state[hist_key]]
            if difficulty == "초급":
                cand = _parse_mc_questions(raw, n_items)
                uniques = [it for it in cand if not any(is_similar(it["q"], pt) for pt in prev_texts)]
                st.session_state[f"{CATEGORY_NAME}_quiz_items"] = uniques
                st.session_state[f"{CATEGORY_NAME}_quiz_mode"]  = "초급"
                st.session_state[hist_key].extend([u["q"] for u in uniques])
            else:
                cand = _extract_questions(raw, n_items)
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
        # 객관식 렌더링 (LLM 채점 불필요, 자체 정답 비교)
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
                # 선택값에서 A/B/C/D 추출
                if sel:
                    sel_letter = sel.split(")")[0]
                else:
                    sel_letter = None
                correct = (sel_letter == it["answer"])
                verdict = "정답" if correct else "오답"
                st.markdown(f"**문항 #{i} 결과**")
                st.markdown(f"**판정: {verdict}**")
                st.markdown(f"피드백: {it.get('expl','(해설 없음)')}")
                st.markdown("---")
    else:
        # 서술형 렌더링 (기존 유지 + LLM 채점)
        for i, qtext in enumerate(items, start=1):
            st.markdown(f"**{i}) {qtext}**")
            st.text_area(
                f"답안 입력 #{i}",
                key=f"{CATEGORY_NAME}_ans_{i-1}",
                height=100,
                placeholder="여기에 본인 답안을 작성하세요."
            )

        # 채점 출력 포맷 확정 (항상 줄바꿈)
        def parse_eval(judged: str):
            s = (judged or "").strip().replace("\r\n", "\n")
            m_verdict = re.search(r"판정\s*:\s*(정답|오답)", s)
            verdict = m_verdict.group(1) if m_verdict else None
            m_feedback = re.search(r"피드백\s*:\s*(.*)", s, flags=re.S)
            feedback = m_feedback.group(1).strip() if m_feedback else None
            if not feedback and m_verdict:
                tail = s.split(m_verdict.group(0), 1)[-1].strip()
                if tail and not tail.lower().startswith("피드백"):
                    feedback = tail
            if not verdict:
                if "정답" in s: verdict = "정답"
                elif "오답" in s: verdict = "오답"
            if not feedback:
                feedback = ""
            return verdict, feedback

        def render_eval(judged: str):
            verdict, feedback = parse_eval(judged)
            st.markdown(f"**판정: {verdict or '판정 불명'}**")
            st.markdown(f"피드백: {feedback or '(없음)'}")

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
                render_eval(judged)
                st.markdown("---")
else:
    st.caption("아직 생성된 문제가 없습니다. ‘랜덤 문제 생성’을 눌러주세요.")