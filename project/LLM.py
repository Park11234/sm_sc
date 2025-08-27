from __future__ import annotations  # ✅ 반드시 파일 최상단, 단 1회

# ── 표준 라이브러리
import os
import io
import re
import base64
import difflib
import tempfile
from typing import List, Tuple, Optional, Iterable

# ── 서드파티 기본
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from faster_whisper import WhisperModel

# ── 선택적 의존성 (설치 여부에 따라 플래그)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

try:
    from langchain_google_genai import (  # type: ignore
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

try:
    from openai import OpenAI as _OpenAIClient  # type: ignore
    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False

# ── LangChain 공용 유틸
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# ── 환경 변수 (.env)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# =========================
# 1) 텍스트 전처리/유사도
# =========================
_STOPWORDS: set[str] = {
    "the","a","an","of","and","to","in","port","on","for","with","by","at","from","is","are","was","were","be","as",
    "및","과","와","에서","으로","으로써","에","의","를","을","은","는","이다","한다","하는","또는",
}

def _normalize_text(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return toks


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
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

# =========================
# 2) PDF → VectorStore(FAISS)
# =========================
class OpenAIEmbeddingsLite:
    """langchain-openai 미설치 환경에서 OpenAI SDK로 임베딩 호출하는 폴백.
    LangChain Embeddings 인터페이스 호환: embed_documents, embed_query
    """
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        if not HAS_OPENAI_SDK:
            raise RuntimeError("OpenAI SDK가 필요합니다. `pip install openai`.")
        key = api_key or OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 또는 환경변수에 설정하세요.")
        self.client = _OpenAIClient(api_key=key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            r = self.client.embeddings.create(model=self.model, input=t)
            out.append(r.data[0].embedding)
        return out

    def embed_query(self, text: str) -> list[float]:
        r = self.client.embeddings.create(model=self.model, input=text)
        return r.data[0].embedding


def build_vectorstore_from_pdfs(
    files: List,  # Streamlit UploadedFile 리스트 또는 파일-유사 객체 리스트
    embed_backend: str = "openai",
):
    """PDF 파일들을 로드해 텍스트를 분할하고, 지정한 임베딩으로 FAISS 인덱스를 생성합니다.

    Args:
        files: 업로드된 PDF 파일 객체들의 리스트. 각 객체는 .read()를 지원해야 합니다.
        embed_backend: "openai" | "gemini"
    Returns:
        FAISS 벡터스토어 인스턴스
    """
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embed_backend = (embed_backend or "openai").lower()

    if embed_backend == "openai":
        if HAS_OPENAI:
            embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""))
        elif HAS_OPENAI_SDK:
            embedding = OpenAIEmbeddingsLite(model="text-embedding-3-small")
        else:
            raise RuntimeError("OpenAI 임베딩 사용 불가: `openai` 또는 `langchain-openai` 설치 필요")
    elif embed_backend == "gemini":
        if not HAS_GEMINI:
            raise RuntimeError("Gemini 임베딩 사용 불가: `langchain-google-genai` 설치 필요")
        embedding = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", ""),
        )
    else:
        raise RuntimeError("지원하는 임베딩 백엔드는 'openai'와 'gemini' 뿐입니다.")

    return FAISS.from_documents(splits, embedding)

# =========================
# 3) DOT 파이프라인 문자열 생성
# =========================

def dot_pipeline(title: str, steps: List[str]) -> str:
    """Graphviz DOT 포맷으로 간단한 파이프라인 그래프를 만듭니다."""
    lines = [
        "digraph G {",
        "rankdir=LR;",
        "node [shape=box, style=rounded, fontsize=12, fontname=\"Pretendard, NanumGothic, Arial\"];",
        f"labelloc=t; label=\"{title}\";",
    ]
    for i, step in enumerate(steps):
        lines.append(f"n{i} [label=\"{step}\"];")
    for i in range(len(steps) - 1):
        lines.append(f"n{i} -> n{i + 1};")
    lines.append("}")
    return "\n".join(lines)

# =========================
# 4) LLM 백엔드 선택 (질의응답 등)
# =========================

def get_llm(backend: str = "openai", model: str = "gpt-4o-mini", temperature: float = 0.2):
    """LangChain Chat 인터페이스(.invoke) 호환 객체 반환.

    backend: "openai" | "gemini"
    """
    b = (backend or "openai").lower()

    if b == "openai":
        if HAS_OPENAI:
            return ChatOpenAI(model=model, temperature=temperature, openai_api_key=OPENAI_API_KEY)
        if not HAS_OPENAI_SDK:
            raise RuntimeError("OpenAI 사용 불가: `openai` 또는 `langchain-openai` 설치 필요")
        # 폴백: OpenAI SDK 래퍼
        client = _OpenAIClient(api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""))

        class _OpenAIChatLite:
            def invoke(self, prompt: str):
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                class _R:
                    pass
                _R.content = r.choices[0].message.content
                return _R()

        return _OpenAIChatLite()

    if b == "gemini":
        if not HAS_GEMINI:
            raise RuntimeError("Gemini 사용 불가: `langchain-google-genai` 설치 필요")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", ""),
        )

    raise RuntimeError("지원하는 LLM 백엔드는 'openai'와 'gemini' 뿐입니다.")


# ===== 사용 예시 =====
if __name__ == "__main__":
    # 1) 유사도 테스트
    a = "이온 주입 공정의 장단점을 설명하라"
    b = "이온주입 공정의 장점과 단점을 논하시오"
    print("similar?", is_similar(a, b))

    # 2) DOT 파이프라인 예시
    print(dot_pipeline("Lithography", ["PR Coat", "Exposure", "Develop"]))

    try:
        import openai  # TTS용
    except Exception:
        openai = None

    # LangChain (LLM 호출용)
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

# --------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------
APP_TITLE = "반도체 공정 Q&A"

SYSTEM_PROMPT = (
    "당신은 반도체 공정(포토, 식각, 증착, CMP, 이온주입, 확산, 열처리 등) 멘토입니다.\n"
    "규칙:\n"
    "- 먼저 의도: 를 한 줄로 제시\n"
    "- 복잡한 주제는 단계 1, 2, ... 형태로 설명\n"
    "- 정보 부족 시 '정보가 부족합니다' 명시\n"
    "- 추론 필요 시 '추론(유형: 연역/귀납/유추): 근거' 1줄 추가\n"
    "- 항상 정중한 한국어(존댓말)로 답변\n"
)

# --------------------------------------------------------------------
# 세션 초기화
# --------------------------------------------------------------------
def init_session_state() -> None:
    """앱에서 사용되는 세션 키들을 초기화합니다."""
    if "history" not in st.session_state:
        st.session_state.history: List[Tuple[str, str]] = []
    if "upload_images" not in st.session_state:
        st.session_state.upload_images: List[Image.Image] = []
    if "camera_images" not in st.session_state:
        st.session_state.camera_images: List[Image.Image] = []
    if "use_upload_for_next" not in st.session_state:
        st.session_state.use_upload_for_next = False
    if "use_camera_for_next" not in st.session_state:
        st.session_state.use_camera_for_next = False

# --------------------------------------------------------------------
# 유틸: 이미지 → data URL (비전 모델 입력용)
# --------------------------------------------------------------------
def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    """PIL 이미지를 data URL로 직렬화합니다."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

# --------------------------------------------------------------------
# STT (Whisper)
# --------------------------------------------------------------------
def load_whisper(model_size: str = "base") -> WhisperModel:
    """faster-whisper 모델 로드(기본: base)."""
    return WhisperModel(model_size, device="auto", compute_type="int8")

def transcribe_audio_bytes(audio_bytes: bytes, model_size: str = "base") -> str:
    """녹음 바이트를 받아 Whisper로 텍스트 변환."""
    if not audio_bytes:
        return ""
    tmp_path = "_tmp_query.wav"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    model = load_whisper(model_size)
    segments, _ = model.transcribe(tmp_path, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", ""))
    return text.strip()

# --------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------
def get_llm(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    """LangChain ChatOpenAI 래퍼 생성."""
    return ChatOpenAI(model=model_name, temperature=0.2)

def ask_llm(query_text: str, images: Optional[List[Image.Image]] = None) -> str:
    """텍스트 + (선택) 다중 이미지로 질의. 시스템 프롬프트는 모듈 기본값 사용."""
    sys_msgs = [SystemMessage(content=SYSTEM_PROMPT)]

    # 사용자 메시지 구성(멀티모달)
    if images:
        content: List[dict] = [{"type": "text", "text": query_text}]
        for img in images:
            data_url = pil_to_data_url(img, fmt="PNG")
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        user_msg = HumanMessage(content=content)
    else:
        user_msg = HumanMessage(content=query_text)

    llm = get_llm()
    response = llm.invoke(sys_msgs + [user_msg])

    # 히스토리 로깅
    st.session_state.history.append(("user", query_text))
    st.session_state.history.append(("assistant", response.content))

    return response.content

# --------------------------------------------------------------------
# TTS + 자동 재생
# --------------------------------------------------------------------
def speak_text(text: str, filename: str = "tts_output.mp3") -> Optional[str]:
    """OpenAI TTS로 음성 파일 생성. 성공 시 파일 경로 반환."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or openai is None:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",  # 환경에 따라 tts-1 계열로 변경 가능
            voice="alloy",
            input=text,
        )
        with open(filename, "wb") as f:
            f.write(resp.read())
        return filename
    except Exception:
        return None

def autoplay_audio_from_file(filepath: str) -> None:
    """브라우저에서 자동 재생(HTML5 <audio>)"""
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        html = f"""
        <audio autoplay playsinline>
            <source src='data:audio/mpeg;base64,{b64}' type='audio/mpeg'>
        </audio>
        """
        components.html(html, height=0)
    except Exception:
        pass

# --------------------------------------------------------------------
# 헬퍼: 다음 질문에 포함할 이미지 선택(카메라, 업로드)
# --------------------------------------------------------------------
def get_selected_images() -> List[Image.Image]:
    """체크 상태에 따라 업로드/카메라 이미지를 합쳐 반환."""
    selected: List[Image.Image] = []
    if st.session_state.use_camera_for_next:
        selected.extend(st.session_state.camera_images)
    if st.session_state.use_upload_for_next:
        selected.extend(st.session_state.upload_images)
    return selected

# STT
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

# TTS (선택)
try:
    import openai
    HAS_OPENAI_SDK = True
except Exception:
    openai = None
    HAS_OPENAI_SDK = False

# LangChain / OpenAI
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LANGCHAIN_OPENAI = True
except Exception:
    HAS_LANGCHAIN_OPENAI = False
    ChatOpenAI = None
    SystemMessage = None
    HumanMessage = None

# Gemini (선택)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_LANGCHAIN_GEMINI = True
except Exception:
    HAS_LANGCHAIN_GEMINI = False

# PDF→VectorStore(필요 시)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    HAS_VS = True
except Exception:
    HAS_VS = False


# ===================== 공용 상수 =====================
APP_TITLE = "반도체 공정 Q&A"
SYSTEM_PROMPT = (
    "당신은 반도체 공정(포토, 식각, 증착, CMP, 이온주입, 확산, 열처리 등) 멘토입니다.\n"
    "규칙:\n"
    "- 먼저 의도: 를 한 줄로 제시\n"
    "- 복잡한 주제는 단계 1, 2, ... 형태로 설명\n"
    "- 정보 부족 시 '정보가 부족합니다' 명시\n"
    "- 추론 필요 시 '추론(유형: 연역/귀납/유추): 근거' 1줄 추가\n"
    "- 항상 정중한 한국어(존댓말)로 답변\n"
)

# ===================== 세션 초기화 =====================
def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history: List[Tuple[str, str]] = []
    if "upload_images" not in st.session_state:
        st.session_state.upload_images: List[Image.Image] = []
    if "camera_images" not in st.session_state:
        st.session_state.camera_images: List[Image.Image] = []
    if "use_upload_for_next" not in st.session_state:
        st.session_state.use_upload_for_next = False
    if "use_camera_for_next" not in st.session_state:
        st.session_state.use_camera_for_next = False

# ===================== 이미지 직렬화 =====================
def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

# ===================== STT (Whisper) =====================
def load_whisper(model_size: str = "base"):
    if not HAS_WHISPER:
        raise RuntimeError("faster-whisper가 필요합니다. `pip install faster-whisper`")
    return WhisperModel(model_size, device="auto", compute_type="int8")

def transcribe_audio_bytes(audio_bytes: bytes, model_size: str = "base") -> str:
    if not audio_bytes:
        return ""
    tmp_path = "_tmp_query.wav"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    model = load_whisper(model_size)
    segments, _ = model.transcribe(tmp_path, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", ""))
    return text.strip()

# ===================== 멀티모달 LLM =====================
def get_llm(model_name: str = "gpt-4o-mini"):
    if not HAS_LANGCHAIN_OPENAI:
        raise RuntimeError("langchain-openai 설치 필요: `pip install langchain-openai`")
    return ChatOpenAI(model=model_name, temperature=0.2)

def ask_llm(query_text: str, images: Optional[List[Image.Image]] = None) -> str:
    if SystemMessage is None or HumanMessage is None:
        raise RuntimeError("langchain-core 설치 필요: `pip install langchain-core`")
    sys_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    if images:
        content: List[dict] = [{"type": "text", "text": query_text}]
        for img in images:
            data_url = pil_to_data_url(img, fmt="PNG")
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        user_msg = HumanMessage(content=content)
    else:
        user_msg = HumanMessage(content=query_text)
    llm = get_llm()
    response = llm.invoke(sys_msgs + [user_msg])
    st.session_state.history.append(("user", query_text))
    st.session_state.history.append(("assistant", response.content))
    return response.content

# ===================== TTS + 자동 재생 =====================
def speak_text(text: str, filename: str = "tts_output.mp3") -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not HAS_OPENAI_SDK:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text,
        )
        with open(filename, "wb") as f:
            f.write(resp.read())
        return filename
    except Exception:
        return None

def autoplay_audio_from_file(filepath: str) -> None:
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        html = f"""
        <audio autoplay playsinline>
            <source src='data:audio/mpeg;base64,{b64}' type='audio/mpeg'>
        </audio>
        """
        components.html(html, height=0)
    except Exception:
        pass

# ===================== 이미지 선택 헬퍼 =====================
def get_selected_images() -> List[Image.Image]:
    selected: List[Image.Image] = []
    if st.session_state.use_camera_for_next:
        selected.extend(st.session_state.camera_images)
    if st.session_state.use_upload_for_next:
        selected.extend(st.session_state.upload_images)
    return selected

# ===================== 유사도(중복 방지) =====================
_STOPWORDS: set[str] = {
    "the","a","an","of","and","to","in","port","on","for","with","by","at","from","is","are","was","were","be","as",
    "및","과","와","에서","으로","으로써","에","의","를","을","은","는","이다","한다","하는","또는",
}
def _normalize_text(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return toks
def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
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

# ===================== 퀴즈/평가 유틸(포토리소그래피 페이지용) =====================
def get_llm_backend():
    """세션에 저장한 백엔드/모델 읽기 (없으면 OpenAI/gpt-4o-mini)."""
    return st.session_state.get("llm_backend", "openai"), st.session_state.get("llm_model", "gpt-4o-mini")

def generate_with_openai(prompt: str, model_name: str) -> str:
    """LangChain 우선, 실패시 OpenAI SDK 폴백."""
    try:
        if not HAS_LANGCHAIN_OPENAI:
            raise RuntimeError("no langchain_openai")
        llm = ChatOpenAI(model=model_name, temperature=0)
        return llm.invoke(prompt).content
    except Exception:
        try:
            if not HAS_OPENAI_SDK:
                raise RuntimeError("no openai sdk")
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"__GEN_ERROR__ {e}"

def generate_with_gemini(prompt: str, model_name: str) -> str:
    """LangChain 우선, 실패시 google-generativeai 폴백."""
    try:
        if not HAS_LANGCHAIN_GEMINI:
            raise RuntimeError("no langchain_google_genai")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        return llm.invoke(prompt).content
    except Exception:
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return getattr(resp, "text", "") or ""
        except Exception as e:
            return f"__GEN_ERROR__ {e}"

def gather_context(k: int = 6, enabled: bool = True, retriever=None) -> str:
    """업로드 문서 컨텍스트 모으기. retriever 없으면 세션의 vectorstore를 사용."""
    if not enabled:
        return ""
    try:
        if retriever is None:
            if "vectorstore" not in st.session_state:
                return ""
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents("핵심 개념 요약")
        return "\n\n".join(d.page_content for d in docs)[:6000]
    except Exception:
        return ""

def extract_questions(s: str, expected_n: int) -> list[str]:
    """서술형 문제 추출(번호 구분/단락 구분/라인 기반)."""
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

def parse_mc_questions(s: str, expected_n: int):
    """LLM 출력 → 객관식 구조화."""
    blocks = re.split(r'^\s*\d+\)\s+', (s or "").strip(), flags=re.M)
    items = []
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
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
                ans = m_ans.group(1).upper(); phase = "ans"; continue
            m_ex  = re.match(r'^해설\s*:\s*(.*)$', ln, flags=re.I)
            if m_ex:
                expl = m_ex.group(1).strip(); phase = "expl"; continue
            if phase == "q":
                q_lines.append(ln)
            elif phase == "expl":
                expl += (" " + ln)
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

def parse_eval(judged: str):
    """채점결과 파싱: (판정, 피드백)"""
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
    """Streamlit로 채점결과 렌더링."""
    verdict, feedback = parse_eval(judged)
    st.markdown(f"**판정: {verdict or '판정 불명'}**")
    st.markdown(f"피드백: {feedback or '(없음)'}")

# ===================== (옵션) PDF→VectorStore =====================
def build_vectorstore_from_pdfs(files: List, embed_backend: str = "openai"):
    """업로드된 PDF들로 FAISS 인덱스 생성 (필요 시 사용)."""
    if not HAS_VS:
        raise RuntimeError("langchain-community 등 벡터스토어 의존성 설치 필요")
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read()); tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())
        finally:
            try: os.remove(tmp_path)
            except Exception: pass
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    # 기본 OpenAI 임베딩 사용
    embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY", ""))
    return FAISS.from_documents(splits, embedding)

def message(content: str, is_user: bool = False, key: str | None = None, avatar_style: str | None = None):
    """
    streamlit_chat.message 대체용.
    추가 설치 없이 st.chat_message로 동일 동작을 흉내냅니다.
    avatar_style, key 인자는 호환성 유지를 위해 받기만 하고 내부에선 사용하지 않습니다.
    """
    role = "user" if is_user else "assistant"
    with st.chat_message(role):
        st.markdown(content)

def hist_pairs(chat_history: list, limit_pairs: int = 6):
    msgs = chat_history or []
    pairs = []
    i = 0
    while i < len(msgs) - 1:
        a, b = msgs[i], msgs[i + 1]
        if a.get("role") == "user" and b.get("role") == "assistant":
            pairs.append((a.get("content", ""), b.get("content", "")))
            i += 2
        else:
            i += 1
    return pairs[-limit_pairs:]


def hist_text(chat_history: list, limit_pairs: int = 6):
    pairs = hist_pairs(chat_history, limit_pairs)
    lines = []
    for u, a in pairs:
        lines.append(f"사용자: {u}\n도우미: {a}")
    return "\n".join(lines)



def summarize_docs(docs, max_chars: int = 2400):
    parts, total = [], 0
    for d in docs:
        txt = (getattr(d, "page_content", "") or "").strip()
        if not txt:
            continue
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "파일")
        pg = meta.get("page", "?")
        frag = f"<<{src} p.{pg}>>\n{txt[:1200]}"
        parts.append(frag)
        total += len(frag)
        if total >= max_chars:
            break
    return "\n\n".join(parts)
# ================= get_chat_llm (OpenAI/Gemini 지원, LangChain ↔ SDK 폴백) =================
def get_chat_llm(backend: str = "openai", model: str = "gpt-4o-mini", temperature: float = 0.2):
    """
    반환값: .invoke(prompt)를 지원하는 객체
      - LangChain이 있으면 ChatOpenAI / ChatGoogleGenerativeAI 인스턴스
      - 없으면 SDK 폴백 래퍼(동일하게 .invoke(prompt) 사용 가능, .invoke 결과는 .content 속성 보유)
    """
    b = (backend or "openai").lower()

    # ---------------- OpenAI ----------------
    if b == "openai":
        # 1) LangChain 우선
        if 'HAS_LANGCHAIN_OPENAI' in globals() and HAS_LANGCHAIN_OPENAI and 'ChatOpenAI' in globals() and ChatOpenAI:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY",""),
            )

        # 2) OpenAI SDK 폴백 (v1 스타일: from openai import OpenAI as _OpenAIClient)
        try:
            from openai import OpenAI as _OpenAIClient  # 이미 상단에서 사용 중인 경로
            _client = _OpenAIClient(api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY",""))
            class _OpenAIChatLite:
                def invoke(self, prompt: str):
                    r = _client.chat.completions.create(
                        model=model,
                        messages=[{"role":"user","content":prompt}],
                        temperature=temperature
                    )
                    class _R: pass
                    _R.content = r.choices[0].message.content
                    return _R()
            return _OpenAIChatLite()
        except Exception:
            pass

        # 3) 구 SDK 폴백 (import openai; openai.ChatCompletion)
        if 'openai' in globals() and openai and hasattr(openai, "ChatCompletion"):
            openai.api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY","")
            class _OpenAIChatLegacy:
                def invoke(self, prompt: str):
                    r = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role":"user","content":prompt}],
                        temperature=temperature
                    )
                    class _R: pass
                    _R.content = r.choices[0].message["content"]
                    return _R()
            return _OpenAIChatLegacy()

        raise RuntimeError("OpenAI 백엔드를 사용할 수 없습니다. 키/패키지를 확인하세요.")

    # ---------------- Gemini ----------------
    if b == "gemini":
        # 1) LangChain 우선
        if 'HAS_LANGCHAIN_GEMINI' in globals() and HAS_LANGCHAIN_GEMINI and 'ChatGoogleGenerativeAI' in globals() and ChatGoogleGenerativeAI:
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY",""),
            )

        # 2) google-generativeai SDK 폴백
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY",""))
            class _GeminiChatLite:
                def __init__(self, model_name: str):
                    self._mdl = genai.GenerativeModel(model_name)
                def invoke(self, prompt: str):
                    r = self._mdl.generate_content(prompt)
                    text = getattr(r, "text", None)
                    if not text and getattr(r, "candidates", None):
                        parts = getattr(r.candidates[0].content, "parts", [])
                        text = getattr(parts[0], "text", "") if parts else ""
                    class _R: pass
                    _R.content = text or ""
                    return _R()
            return _GeminiChatLite(model)
        except Exception as e:
            raise RuntimeError(f"Gemini 백엔드를 사용할 수 없습니다: {e}")

    # ---------------- 기타 ----------------
    raise RuntimeError("지원하는 LLM 백엔드는 'openai' 또는 'gemini'입니다.")


