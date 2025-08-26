import os
import streamlit as st
import re
import difflib

# =============================================
# File: pages/1_포토리소그래피.py
# =============================================

from langchain.chains import RetrievalQA


st.set_page_config(page_title="포토리소그래피", layout="wide")

# ---------------- 유사도 판정 유틸 ----------------
_STOPWORDS = {"the","a","an","of","and","to","in","port","on","for","with","by","at","from","is","are","was","were","be","as",
              "및","과","와","에서","으로","으로써","에","의","를","을","은","는","이다","한다","하는","또는"}

def _normalize_text(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return toks

def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def is_similar(q: str, p: str, jaccard_thr: float = 0.55, ratio_thr: float = 0.70) -> bool:
    ta, tb = _normalize_text(q), _normalize_text(p)
    if _jaccard(ta, tb) >= jaccard_thr:
        return True
    if difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio() >= ratio_thr:
        return True
    return False

# ---------------- 페이지 본문 ----------------
st.header("1) 포토리소그래피")

# 개요
st.subheader("개요")
st.write("웨이퍼 표면에 감광막을 바르고 노광·현상으로 패턴을 형성합니다.")

# 핵심 포인트 (툴팁)
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
# 진도 상태 초기화
if "progress" not in st.session_state:
    st.session_state.progress = {step["name"]: False for step in steps_data}

# 단계별 설명 및 체크박스
completed = 0
for step in steps_data:
    with st.expander(f"{step['icon']} {step['name']}"):
        st.write(step["desc"])
        checked = st.checkbox("이 단계 학습 완료", value=st.session_state.progress[step["name"]], key=step["name"])
        st.session_state.progress[step["name"]] = checked
        if checked:
            completed += 1

# 전체 진도율 표시
total = len(steps_data)
percent = int((completed / total) * 100)
st.progress(percent)
st.caption(f"📘 학습 진도: {completed} / {total} 단계 완료 ({percent}%)")

# 다음 단계 안내
if completed < total:
    next_step = [s for s in steps_data if not st.session_state.progress[s["name"]]][0]
    st.info(f"다음 추천 학습 단계: {next_step['name']}")
else:
    st.success("🎉 모든 단계를 완료했습니다! 복습하거나 질의응답을 활용해보세요.")


# ---------------- 질의응답 (RAG) ----------------
st.subheader("질의응답 (RAG)")
if "vectorstore" not in st.session_state:
    st.info("임베딩 자료가 없습니다. 메인에서 PDF 업로드 → 임베딩 생성 후 이용하세요.")
else:
    if "qa_chain" not in st.session_state:
        backend = st.session_state.get("llm_backend", "openai")
        model = st.session_state.get("llm_model", "gpt-4o-mini")
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        if backend == "openai":
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=model, temperature=0.2)
            except Exception:
                # 폴백: OpenAI SDK (선택)
                from openai import OpenAI
                _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",""))
                class _OpenAILLM:
                    def invoke(self, prompt):
                        r = _client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], temperature=0.2)
                        class _R: content = r.choices[0].message.content
                        return _R()
                llm = _OpenAILLM()
        else:
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model=model, temperature=0.2)

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )

    q = st.text_input("질문을 입력하세요", placeholder="예: EUV와 DUV 차이")
    if st.button("질문하기", use_container_width=True):
        if q.strip():
            out = st.session_state.qa_chain({"query": q})
            st.markdown("### 답변")
            st.write(out.get("result", "정보가 부족합니다"))
            src = out.get("source_documents") or []
            if src:
                with st.expander("출처"):
                    for i, sdoc in enumerate(src, 1):
                        meta = sdoc.metadata or {}
                        st.write(f"{i}. {meta.get('source', '파일')} p.{meta.get('page', '?')}")
        else:
            st.warning("질문을 입력하세요.")

# ---------------- 랜덤 문제 생성기 + 채점 ----------------
st.subheader("랜덤 문제 생성기")
CATEGORY_NAME = "포토리소그래피"  # ← 페이지 주제명

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