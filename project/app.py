import os

import streamlit as st



page_main = st.Page("main.py", title="main Page", icon="📟")
page_1 = st.Page("cate.py", title="목차", icon="📟")
page_2 = st.Page("1.py", title="1.포토리소그래피", icon="📟")
page_3 = st.Page("2.py", title="2.식각(Etch)", icon="📟")
page_4 = st.Page("3.py", title="3.산화 (Oxidation)", icon="📟")
page_5 = st.Page("4.py", title="4.확산 (Diffusion)", icon="📟")
page_6 = st.Page("5.py", title="5.이온주입 (Ion Implantation)", icon="📟")
page_7 = st.Page("6.py", title="6.증착 (CVD/PVD/ALD)", icon="📟")
page_8 = st.Page("7.py", title="7.금속배선 (Metallization)", icon="📟")
page_9 = st.Page("8.py", title="8.평탄화 (CMP)", icon="📟")
page_10 = st.Page("9.py", title="9.질문(임시요)", icon="📟")


page = st.navigation([page_main,page_1,page_2,page_3,page_4,page_5,page_6,page_7,page_8,page_9])

page.run()
