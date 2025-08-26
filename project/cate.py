import os
import tempfile
from typing import List
import streamlit as st

# LangChain / Vector DB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

st.title("목차")
st.markdown("1.포토리소그래피")
st.markdown("2.식각(Etch)")
st.markdown("3.산화 (Oxidation)")
st.markdown("4.확산 (Diffusion)")
st.markdown("5.이온주입 (Ion Implantation)")
st.markdown("6.증착 (CVD/PVD/ALD)")
st.markdown("7.금속배선 (Metallization)")
st.markdown("8.평탄화 (CMP)")

st.markdown("임시 디자인")
