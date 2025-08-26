import os
import tempfile
from typing import List
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoaderimport
from langchain_google_genai import ChatGoogleGenerativeAI
import time
import urllib
import google.generativeai as genai
from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI

def build_vectorstore_from_pdfs(files: List[st.runtime.uploaded_file_manager.UploadedFile],
                                embed_backend: str = "openai",
                                ollama_embed_model: str = "nomic-embed-text"):
    docs = []
    with st.spinner("PDF를 로딩 중…"):
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            docs.extend(loader.load())
            os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    if embed_backend == "openai":
        if not HAS_OPENAI:
            raise RuntimeError("langchain-openai가 필요합니다.")
        embedding = OpenAIEmbeddings()
    elif embed_backend == "ollama":
        if not HAS_OLLAMA:
            raise RuntimeError("Ollama 임베딩 사용 불가 (설치 필요)")
        embedding = OllamaEmbeddings(model=ollama_embed_model)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vs = FAISS.from_documents(splits, embedding)
    return vs


def set_llm_settings():
    st.sidebar.subheader("LLM · 임베딩 설정")
    backend = st.sidebar.selectbox("LLM 백엔드", ["openai", "ollama"], index=0)
    if backend == "openai":
        if HAS_OPENAI:
            _api = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-…")
            if _api:
                os.environ["OPENAI_API_KEY"] = _api
            model = st.sidebar.text_input("OpenAI 모델", value="gpt-4o-mini")
        else:
            st.sidebar.error("langchain-openai 미설치")
            model = "gpt-4o-mini"
    else:
        if HAS_GEMINI:
            model = st.sidebar.text_input("Gemini 모델", value="llama3.1:8b")
        else:
            st.sidebar.error("Ollama 미설치")
            model = "llama3.1:8b"

    st.session_state["llm_backend"] = backend
    st.session_state["llm_model"] = model

    st.sidebar.divider()
    st.sidebar.subheader("자료 업로드 · 임베딩")
    uploaded = st.sidebar.file_uploader("PDF 업로드 (여러 개)", type=["pdf"], accept_multiple_files=True)
    embed_backend = st.sidebar.selectbox("임베딩 백엔드", ["openai", "ollama", "hf"], index=0)

    colA, colB = st.sidebar.columns(2)
    if colA.button("임베딩 생성", use_container_width=True):
        if not uploaded:
            st.sidebar.warning("PDF를 먼저 업로드하세요.")
        else:
            try:
                st.session_state.vectorstore = build_vectorstore_from_pdfs(uploaded, embed_backend)
                st.sidebar.success("벡터스토어 생성 완료")
                st.session_state.pop("qa_chain", None)
            except Exception as e:
                st.sidebar.error(f"임베딩 실패: {e}")

    if colB.button("임베딩 초기화", use_container_width=True):
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("qa_chain", None)
        st.sidebar.info("임베딩을 비웠습니다.")


def dot_pipeline(title: str, steps):
    lines = ["digraph G {",
             "rankdir=LR;",
             "node [shape=box, style=rounded, fontsize=12, fontname=\"Pretendard, NanumGothic, Arial\"];",
             f"labelloc=t; label=\"{title}\";"]
    for i, step in enumerate(steps):
        lines.append(f"n{i} [label=\"{step}\"];")
    for i in range(len(steps) - 1):
        lines.append(f"n{i} -> n{i + 1};")
    lines.append("}")
    return "\n".join(lines)

d_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

def openAiModel():
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

def makeMsg(system,user ):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages

def openAiModelArg(model, msgs):
    print(model)
    print(msgs)
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=msgs
    )
    return response.choices[0].message.content


def geminiModel():
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model

def geminiTxt(txt):
    model = geminiModel()
    response = model.generate_content(txt)
    return response.text

def openAiModel():
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

# OpenAI LLM Model
def getOpenAI():
    llm = ChatOpenAI(temperature=0, model_name='gpt-4o')
    return llm

# Gemini LLM Model
def getGenAI():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_output_tokens=200,
        google_api_key=GOOGLE_API_KEY
    )
    return llm


def save_carpturefile(directory, picture, name, st):
    if picture is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 2. 파일 저장 (이름 변경 없이 저장)
        with open(os.path.join(directory, name), 'wb') as file:
            file.write(picture.getvalue())
        # 3. 저장 완료 메시지 출력
        st.success(f'저장 완료: {directory}에 {name} 저장되었습니다.')

def save_uploadedfile(directory, file, st):
    # 1. 디렉토리가 없으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 2. 파일 저장 (이름 변경 없이 저장)
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())
    # 3. 저장 완료 메시지 출력
    st.success(f'저장 완료: {directory}에 {file.name} 저장되었습니다.')


def progressBar(txt):
    # Progress Bar Start -----------------------------------------
    progress_text = txt
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.08)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    return my_bar
    # Progress Bar End -----------------------------------------

def makeAudio(text, name):
    if not os.path.exists("audio"):
        os.makedirs("audio")
    model = openAiModel()
    response = model.audio.speech.create(
        model="tts-1",
        input=text,
        #["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        voice="alloy",
        response_format="mp3",
        speed=1.1,
    )
    response.stream_to_file("audio/"+name)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def makeImage(prompt, name):
    openModel = openAiModel()
    response = openModel.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    print(image_url)
    imgName = "img/"+name
    urllib.request.urlretrieve(image_url,  imgName)

def makeImages(prompt, name, num):
    openModel = openAiModel()
    response = openModel.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="1024x1024",
        n=num,
    )
    for n,data in enumerate(response.data):
        print(n)
        print(data.url)
        imgname = f"img/{name.split('.')[0]}_{n}.png"
        urllib.request.urlretrieve(data.url, imgname)

def cloneImage(imgName, num):
    openModel = openAiModel()
    response = openModel.images.create_variation(
        model="dall-e-2",
        image=open("img/"+imgName, "rb"),
        n=num,
        size="1024x1024"
    )
    for n,data in enumerate(response.data):
        print(n)
        print(data.url)
        name = f"img/{imgName.split('.')[0]}_clone_{n}.png"
        urllib.request.urlretrieve(data.url, name)