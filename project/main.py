# =============================================
# File: app.py  (메인 페이지)
# =============================================
import os
import tempfile
from typing import List
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# --- LangChain용 OpenAI(있으면 사용) ---
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# --- Gemini ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# --- OpenAI 공식 SDK (폴백용) ---
try:
    from openai import OpenAI as _OpenAIClient
    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False

from langchain.chains import RetrievalQA

icon_path = "electronic_technology_chip_cpu_processor_icon_261973.png"
st.set_page_config(
    page_title="반도체 공정 학습 튜터 · 메인",
)

# ---- OpenAI 임베딩 폴백 클래스 (langchain-openai 미설치 시) ----
class OpenAIEmbeddingsLite:
    """
    langchain-openai 없이 OpenAI SDK로 임베딩 호출.
    LangChain Embeddings 인터페이스(embed_documents, embed_query)만 구현.
    """
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY가 없습니다. 사이드바에서 입력하거나 환경변수로 설정하세요.")
        self.client = _OpenAIClient(api_key=key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            resp = self.client.embeddings.create(model=self.model, input=t)
            vecs.append(resp.data[0].embedding)
        return vecs

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

# --- Helper: PDF → VectorStore(FAISS) ---
def build_vectorstore_from_pdfs(files, embed_backend: str = "openai"):
    import tempfile, os
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS

    docs = []
    with st.spinner("PDF를 로딩 중…"):
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                loader = PyPDFLoader(tmp_path)
                docs.extend(loader.load())
            finally:
                os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    if embed_backend == "openai":
        if HAS_OPENAI:
            embedding = OpenAIEmbeddings()
        elif HAS_OPENAI_SDK:
            # 폴백: OpenAI SDK로 임베딩 호출
            embedding = OpenAIEmbeddingsLite(model="text-embedding-3-small")
        else:
            raise RuntimeError("OpenAI 임베딩 사용 불가: openai SDK 설치 필요 (`pip install openai`)")
    elif embed_backend == "gemini":
        if not HAS_GEMINI:
            raise RuntimeError("Gemini 임베딩 사용 불가: langchain-google-genai 설치 필요")
        embedding = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    return FAISS.from_documents(splits, embedding)

# --- Sidebar: LLM & Embeddings 설정 + PDF 임베딩 ---
st.sidebar.subheader("LLM · 임베딩 설정")

backend = st.sidebar.selectbox("LLM 백엔드", ["openai", "gemini"], index=0)

if backend == "openai":
    # langchain-openai 유무와 관계없이 키는 필요
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if HAS_OPENAI:
        model = st.sidebar.text_input("OpenAI 모델", value="gpt-4o")
    else:
        model = st.sidebar.text_input("OpenAI 모델", value="gpt-4o")
else:
    if HAS_GEMINI:
        google_key = st.sidebar.text_input("Google API Key", type="password", placeholder="AIza...")
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key
        model = st.sidebar.text_input("Gemini 모델", value="gemini-1.5-flash")
    else:
        st.sidebar.error("langchain-google-genai 미설치 (`pip install langchain-google-genai`)")
        model = "gemini-1.5-flash"

st.session_state["llm_backend"] = backend
st.session_state["llm_model"] = model

st.sidebar.divider()
st.sidebar.subheader("자료 업로드 · 임베딩")

embed_backend = st.sidebar.selectbox("임베딩 백엔드", ["openai", "gemini", "hf"], index=0)

uploaded = st.sidebar.file_uploader("PDF 업로드 (여러 개 가능)", type=["pdf"], accept_multiple_files=True)

colA, colB = st.sidebar.columns(2)
if st.sidebar.button("생성", use_container_width=True):
    if not uploaded:
        st.sidebar.warning("PDF를 먼저 업로드하세요.")
    else:
        try:
            st.session_state.vectorstore = build_vectorstore_from_pdfs(uploaded, embed_backend)
            st.sidebar.success("벡터스토어 생성 완료")
            st.session_state.pop("qa_chain", None)
        except Exception as e:
            st.sidebar.error(f"임베딩 실패: {e}")

if st.sidebar.button("초기화", use_container_width=True):
    st.session_state.pop("vectorstore", None)
    st.session_state.pop("qa_chain", None)
    st.sidebar.info("임베딩을 비웠습니다.")

st.title("반도체 공정 학습 튜터")
st.markdown("임시 디자인")
