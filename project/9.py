import os
import tempfile
from typing import List
import streamlit as st

# LangChain / Vector DB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# =============================================
# File: pages/8_CMP.py
# =============================================
import streamlit as st
from langchain.chains import RetrievalQA

st.set_page_config(page_title="CMP", layout="wide")

st.header("8) 평탄화 (CMP)")

st.subheader("개요")
st.write("연마 패드와 슬러리를 사용해 표면을 평탄화하는 공정입니다.")

st.subheader("핵심 포인트")
st.markdown("""
- 패드/슬러리/다운포스/상대속도 제어
- 디싱·스크래치·오버폴리시 제어, 엔드포인트 관리
""")

st.subheader("프로세스 다이어그램")
steps = ["Pad Conditioning", "Polish (Slurry)", "Endpoint Ctrl", "Clean", "Metrology"]
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

    q = st.text_input("질문을 입력하세요", placeholder="예: 디싱 원인")
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
