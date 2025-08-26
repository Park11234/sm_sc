import streamlit as st

page_main = st.Page("main.py", title="main Page", icon="📟")
page_1 = st.Page("1.py", title="Page1", icon="📟")
page_2 = st.Page("2.py", title="Page2", icon="📟")
page_3 = st.Page("3.py", title="Page3", icon="📟")
page_4 = st.Page("4.py", title="Page4", icon="📟")
page_5 = st.Page("5.py", title="Page5", icon="📟")
page_6 = st.Page("6.py", title="Page6", icon="📟")
page_7 = st.Page("7.py", title="Page7", icon="📟")
page_8 = st.Page("8.py", title="Page8", icon="📟")
page_9 = st.Page("9.py", title="Page9", icon="📟")

page = st.navigation([page_main,page_1,page_2,page_3,page_4,page_5,page_6,page_7,page_8,page_9])

page.run()