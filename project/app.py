import streamlit as st
from LLM import build_vectorstore_from_pdfs

uploaded = st.sidebar.file_uploader("PDF 업로드 (여러 개)", type=["pdf"], accept_multiple_files=True)
embed_backend = st.sidebar.selectbox("임베딩 백엔드", ["openai", "gemini"], index=0)

if st.sidebar.button("임베딩 생성", use_container_width=True):
    if not uploaded:
        st.sidebar.warning("PDF를 먼저 업로드하세요.")
    else:
        try:
            st.session_state.vectorstore = build_vectorstore_from_pdfs(uploaded, embed_backend)
            st.sidebar.success("벡터스토어 생성 완료")
            st.session_state.pop("qa_chain", None)
        except Exception as e:
            st.sidebar.error(f"임베딩 실패: {e}")

page_main = st.Page("main.py", title="main Page", icon="🖥️")
page_2 = st.Page("1.py", title="1.포토리소그래피", icon="📟")
page_3 = st.Page("2.py", title="2.식각(Etch)", icon="📟")
page_4 = st.Page("3.py", title="3.산화 (Oxidation)", icon="📟")
page_5 = st.Page("4.py", title="4.확산 (Diffusion)", icon="📟")
page_6 = st.Page("5.py", title="5.이온주입 (Ion Implantation)", icon="📟")
page_7 = st.Page("6.py", title="6.증착 (CVD/PVD/ALD)", icon="📟")
page_8 = st.Page("7.py", title="7.금속배선 (Metallization)", icon="📟")
page_9 = st.Page("8.py", title="8.평탄화 (CMP)", icon="📟")
page_10 = st.Page("9.py", title="9.접근성+", icon="🗣️")


page = st.navigation([page_main,page_2,page_3,page_4,page_5,page_6,page_7,page_8,page_9, page_10])

page.run()
