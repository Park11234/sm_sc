import os
import tempfile
from typing import List
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# =============================================
# File: pages/1_포토리소그래피.py
# =============================================
import streamlit as st
from langchain.chains import RetrievalQA

st.set_page_config(page_title="포토리소그래피", layout="wide")

st.header("1) 포토리소그래피")

st.subheader("개요")
st.write("웨이퍼 표면에 감광막을 바르고 노광·현상으로 패턴을 형성합니다.")

st.subheader("핵심 포인트")
st.markdown("""
- PR 코팅 → 소프트베이크 → 노광(EUV/DUV) → PEB → 현상 → 하드베이크 → 검사
- 해상도(λ, NA, k1), 포커스/도즈, LER/LWR
""")

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
