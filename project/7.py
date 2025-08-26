import os
import tempfile
from typing import List
import streamlit as st

# LangChain / Vector DB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# =============================================
# File: pages/6_증착.py
# =============================================
import streamlit as st
from langchain.chains import RetrievalQA

st.set_page_config(page_title="증착", layout="wide")

st.header("6) 증착 (CVD/PVD/ALD)")

st.subheader("개요")
st.write("웨이퍼 표면에 절연/도전/배리어 등 필요한 박막을 증착합니다.")

st.subheader("핵심 포인트")
st.markdown("""
- PVD(스퍼터/증발), CVD(LPCVD/PECVD), ALD
- 균일도/응력/결함/접착력 관리
""")

st.subheader("프로세스 다이어그램")
steps = ["Pre-Clean", "PVD/CVD/ALD", "Post-Treatment", "Metrology"]
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

    q = st.text_input("질문을 입력하세요", placeholder="예: ALD의 장점")
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
