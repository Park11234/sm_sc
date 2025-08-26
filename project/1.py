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

# 공정 단계 설명 + 진도 관리
st.subheader("공정 단계 설명 및 진도 관리")

# 단계 데이터
steps_data = [
    {
        "name": "웨이퍼 세정 (Wafer Clean)",
        "desc": """
🧼 **설명: 웨이퍼 세정(Wafer Clean)**

웨이퍼 세정은 반도체 공정의 시작이자 품질을 좌우하는 핵심 단계입니다. 단순한 '청소'가 아닙니다.  
이 단계에서 오염물 하나, 파티클 하나가 수율을 떨어뜨리고 불량을 유발할 수 있습니다.

**왜 세정이 중요한가?**
- 감광막 도포 전, 표면이 완벽히 깨끗해야 패턴이 정확히 형성됩니다.
- 오염물은 PR 도포 불균일, 노광 오류, 식각 불량을 유발합니다.

**제거 대상**
- 유기물 (PR 잔류물, 사람 손 등)
- 금속 이온 (Cu, Fe 등)
- 파티클 (먼지, 미세 입자)

**세정 방식**
- RCA 세정: 화학 용액으로 유기물·금속 제거
- DIW 세정: 초순수로 물리적 세정
- 플라즈마 세정: 산소 플라즈마로 유기물 분해

**세정 후 확인**
- AFM/SEM으로 표면 분석
- 드라이 공정으로 수분 제거
- 클린룸 관리로 재오염 방지

🎯 핵심 요약:
- 웨이퍼 세정은 공정의 품질 기준선
- RCA 세정은 시험 단골
- 세정 실패는 공정 전체 실패로 이어짐
""",
        "icon": "🧼"
    },

    {
        "name": "감광막 도포 (PR Coat)",
        "desc": """
    🧴 **강사 설명: 감광막 도포 (Photoresist Coating)**

    감광막(PR)은 빛에 반응하는 특수 화학물질로, 노광 시 마스크 패턴을 웨이퍼에 전사할 수 있게 해주는 핵심 재료입니다.  
    이걸 균일하게 도포하지 않으면 이후 노광, 현상, 식각 공정에서 **패턴 왜곡, 해상도 저하, 불량 발생**이 일어납니다.

    🔍 **왜 도포하는가?**
    - PR은 빛에 반응하여 패턴을 형성할 수 있는 기반층
    - 노광 시 마스크 패턴을 정확히 전사하기 위한 준비 단계

    🌀 **도포 방식: Spin Coating**
    - 웨이퍼를 고속 회전시켜 PR을 균일하게 퍼뜨리는 방식
    - PR의 **점도**, **회전 속도(rpm)**, **도포 시간**이 두께와 균일성에 영향

    🔥 **도포 후 바로 Soft Bake**
    - PR 내 용매를 증발시켜 안정화
    - Bake를 하지 않으면 노광 시 패턴 번짐 발생

    ⚠️ **주의사항**
    - 중심부와 가장자리 두께 차이 → 균일성 확보 중요
    - PR 오염 방지 → 클린룸 관리 철저
    - PR 종류: Positive vs Negative → 노광 후 제거 방식 다름

    🎯 **핵심 요약**
    - 감광막은 패턴 형성의 기반층
    - Spin Coating은 속도·점도·시간이 핵심
    - 도포 후 Soft Bake는 필수
    - PR 균일성은 해상도 품질을 좌우
    """,
        "icon": "🧴"
    },

    {
        "name": "소프트 베이크 (Soft Bake)",
        "desc": """
    🔥 **강사 설명: 소프트 베이크 (Soft Bake)**

    감광막(PR)을 도포한 직후에는 내부에 용매가 남아 있어, 노광 시 빛이 번지고 패턴 품질이 저하됩니다.  
    소프트 베이크는 이 용매를 증발시켜 PR을 안정화시키는 핵심 단계입니다.

    🧪 **Bake 조건**
    - 온도: 90~110°C (너무 높으면 PR 경화)
    - 시간: 수십 초~수 분 (PR 종류에 따라 조절)
    - 장비: Hot Plate 또는 Oven (Hot Plate가 더 균일)

    ⚠️ **주의사항**
    - 과도한 Bake → PR 경화로 노광 반응 저하
    - 불충분한 Bake → 용매 잔류로 패턴 흐림
    - 급속 냉각 → 균열 발생 가능

    🎯 **핵심 요약**
    - 소프트 베이크는 PR 안정화의 핵심
    - Bake 조건이 패턴 품질을 좌우
    - 실패 시 노광 품질 저하
    """,
        "icon": "🔥"
    },
    {
        "name": "노광 (Exposure)",
        "desc": """
    💡 **강사 설명: 노광 (Exposure)**

    노광은 감광막(PR)에 빛을 조사하여 마스크 패턴을 전사하는 공정입니다.  
    이때 사용하는 빛의 종류와 조사 조건이 패턴의 정밀도와 해상도를 결정합니다.

    🔬 **해상도 공식**
    해상도 ≈ k₁ × (λ / NA)
    - λ: 파장 (짧을수록 좋음) → EUV(13.5nm) > DUV(193nm)
    - NA: 개구수 (클수록 좋음)
    - k₁: 공정 계수 (작을수록 좋음)

    🧪 **노광 조건**
    - 포커스(Focus): 초점 정확도
    - 도즈(Dose): 빛의 에너지량
    - LER/LWR: 패턴의 선(edge) 거칠기

    ⚠️ **주의사항**
    - 마스크 정렬 오차 → 패턴 위치 오류
    - 진동, 온도 변화 → 포커스 불안정
    - PR 종류에 따라 반응 민감도 다름

    🎯 **핵심 요약**
    - 노광은 패턴 정밀도와 해상도를 결정하는 핵심 공정
    - EUV는 더 미세한 패턴 구현 가능
    - 포커스·도즈·LER/LWR이 패턴 품질을 좌우
    """,
        "icon": "💡"
    },
    {
        "name": "PEB (Post-Exposure Bake)",
        "desc": """
    ♨️ **설명: PEB (Post-Exposure Bake)**

    PEB는 노광 후 감광막(PR)에 발생한 화학 반응을 안정화시키는 열처리 공정입니다.  
    특히 Chemically Amplified Resist(CAR)를 사용하는 경우, 산 생성과 확산이 패턴 품질을 좌우합니다.

    🔬 **PEB의 역할**
    - 산 확산 제어 → PR 반응 유도
    - LER 개선 → 패턴 경계 선명도 향상
    - 해상도 향상 → Bake 조건에 따라 결정

    🧪 **Bake 조건**
    - 온도: 90~130°C
    - 시간: 수십 초~수 분
    - 장비: Hot Plate 사용

    ⚠️ **주의사항**
    - 과도한 Bake → 산 확산 과다로 패턴 번짐
    - 부족한 Bake → 반응 불완전
    - 급속 냉각 → PR 균열 가능

    🎯 **핵심 요약**
    - PEB는 노광 후 화학 반응 안정화 공정
    - 산 확산이 패턴 품질을 좌우
    - Bake 조건이 해상도와 수율에 영향
    """,
        "icon": "♨️"
    },
    {
        "name": "현상 (Develop)",
        "desc": """
    🧪 **일타강사 설명: 현상 (Develop)**

    현상은 노광된 감광막(PR) 영역을 현상액으로 제거하여 실제 패턴을 형성하는 공정입니다.  
    이 단계에서 패턴이 눈에 보이는 구조로 완성되며, PR 종류에 따라 제거되는 영역이 달라집니다.

    🔍 **Positive vs Negative PR**
    - Positive PR: 노광된 영역이 제거됨
    - Negative PR: 노광된 영역이 남음

    🧪 **현상 조건**
    - 현상액 농도: 과현상/미현상 방지
    - 시간: PR 두께에 따라 조절
    - 온도: 일정 유지로 반응 안정화

    ⚠️ **주의사항**
    - 과현상 → 패턴 손상
    - 미현상 → 패턴 미형성
    - 현상 후 세정 → 잔류물 제거 필수

    🎯 **핵심 요약**
    - 현상은 패턴을 실제로 형성하는 핵심 공정
    - PR 종류에 따라 제거 방식 달라짐
    - 조건이 패턴 품질을 좌우
    """,
        "icon": "🧪"
    },

    {
        "name": "하드 베이크 (Hard Bake)",
        "desc": """
    🧱 **일타강사 설명: 하드 베이크 (Hard Bake)**

    하드 베이크는 현상 후 형성된 패턴을 고온에서 열처리하여 구조를 고정하고, 이후 공정에서 물리·화학적 손상에 견딜 수 있도록 강화하는 단계입니다.

    🔬 **하드 베이크의 목적**
    - 패턴 고정 → PR 경화로 구조 안정화
    - 내열성 향상 → 고온 공정에서도 패턴 유지
    - 내화학성 강화 → 식각액, 플라즈마 등 공격에 대한 저항성 증가

    🧪 **Bake 조건**
    - 온도: 120~150°C
    - 시간: 수 분 이상
    - 장비: Hot Plate 또는 Oven

    ⚠️ **주의사항**
    - 과도한 Bake → PR 갈변, 패턴 손상
    - 부족한 Bake → 내열성·내화학성 부족
    - 급속 냉각 → 균열 발생 가능

    🎯 **핵심 요약**
    - 하드 베이크는 패턴을 고정하고 보호하는 열처리 공정
    - Bake 조건이 패턴 안정성과 내구성을 결정
    - 이후 공정에서 패턴 손상을 방지하는 방패 역할
    """,
        "icon": "🧱"
    },

    {
        "name": "검사 (Inspection)",
        "desc": """
    🔍 **일타강사 설명: 검사 (Inspection)**

    검사는 포토리소그래피 공정의 마지막 단계로, 형성된 패턴의 품질을 확인하고 다음 공정으로 넘길지를 결정하는 품질 게이트입니다.

    🧪 **검사 항목**
    - 정렬 정확도: 마스크 패턴 위치 확인
    - 결함 검사: 누락, 과현상, 잔류물 등
    - 치수 측정(CD): 패턴 선폭이 설계값과 일치하는지
    - 표면 거칠기(LER): 엣지 품질 평가

    🔬 **검사 장비**
    - SEM, CD-SEM, Overlay Tool, AOI 등

    ⚠️ **주의사항**
    - 검사 누락 → 불량 웨이퍼 통과
    - 과도한 검사 → 시간·비용 증가
    - 기준 미설정 → 불량 판단 모호

    🎯 **핵심 요약**
    - 검사는 패턴 품질을 최종 확인하는 품질 게이트
    - 검사 결과는 공정 피드백으로도 활용됨
    - 정렬·결함·치수·거칠기 등 다각도 평가 필수
    """,
        "icon": "🔍"
    }

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