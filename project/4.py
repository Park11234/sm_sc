import os
import tempfile
from typing import List
import streamlit as st

# LangChain / Vector DB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader