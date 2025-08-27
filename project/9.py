import os
from typing import List
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from PIL import Image
from LLM import (
    APP_TITLE,
    init_session_state,
    transcribe_audio_bytes,
    ask_llm,
    speak_text,
    autoplay_audio_from_file,
    get_selected_images,
)

# 세션 준비
init_session_state()

# ===== UI 시작 =====
st.title(APP_TITLE)

# 1️⃣ 음성 인식 (자동 질문/응답/TTS)
with st.container():
    st.markdown("### 1️⃣ 음성 인식 (자동 질문/응답)")
    st.caption("말하고 멈추면 자동으로 문장 변환 → 답변 생성 → 선택 시 음성으로 재생합니다.")
    audio_bytes = audio_recorder(text="말하기")
    tts_auto = st.checkbox("음성 답변 자동 재생", value=True)

    if audio_bytes:
        text_in = transcribe_audio_bytes(audio_bytes)
        if text_in:
            st.success(f"인식된 질문: {text_in}")
            images = get_selected_images()
            with st.spinner("응답 생성 중..."):
                answer = ask_llm(text_in, images=images)
            st.markdown("#### 답변")
            st.markdown(answer)
            if tts_auto:
                fn = speak_text(answer)
                if fn and os.path.exists(fn):
                    autoplay_audio_from_file(fn)

# 2️⃣ 이미지 업로드 (여러 장)
with st.container():
    st.markdown("### 2️⃣ 이미지 업로드")
    files = st.file_uploader(
        "공정 사진/도표 여러 장 업로드",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    if files:
        imgs: List[Image.Image] = []
        for f in files:
            try:
                img = Image.open(f).convert("RGB")
                imgs.append(img)
            except Exception:
                continue
        if imgs:
            st.session_state.upload_images = imgs
            st.image(imgs, caption=[f"업로드 {i+1}" for i in range(len(imgs))])

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.session_state.use_upload_for_next = st.checkbox("이번 질문에 업로드 이미지 포함")
    with col_u2:
        if st.button("업로드 이미지 비우기"):
            st.session_state.upload_images = []
            st.success("업로드 이미지 목록을 비웠습니다.")

# 3️⃣ 카메라 촬영 (여러 장)
with st.container():
    st.markdown("### 3️⃣ 카메라 촬영")
    cam = st.camera_input("카메라로 촬영")
    if cam is not None:
        try:
            imgc = Image.open(cam).convert("RGB")
            st.session_state.camera_images.append(imgc)
            st.success("카메라 이미지가 목록에 추가되었습니다.")
        except Exception:
            st.warning("카메라 이미지를 불러오지 못했습니다.")

    if st.session_state.camera_images:
        st.image(
            st.session_state.camera_images,
            caption=[f"카메라 {i+1}" for i in range(len(st.session_state.camera_images))]
        )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.session_state.use_camera_for_next = st.checkbox("이번 질문에 카메라 이미지 포함")
    with col_c2:
        if st.button("카메라 이미지 비우기"):
            st.session_state.camera_images = []
            st.success("카메라 이미지 목록을 비웠습니다.")
    st.caption("두 체크 모두 켜면 카메라 이미지가 먼저, 업로드 이미지가 다음으로 함께 전송됩니다.")

# 4️⃣ 채팅 입력 (수동)
with st.container():
    st.markdown("### 4️⃣ 채팅 입력")
    user_text = st.text_area("반도체 공정에 대해 궁금한 점을 입력하십시오.", height=120)
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        send_btn = st.button("질문 보내기", type="primary")
    with col_s2:
        tts_toggle = st.checkbox("답변을 음성으로 재생", value=False)

    if send_btn:
        if not user_text.strip():
            st.warning("질문이 비어 있습니다.")
        else:
            images = get_selected_images()
            with st.spinner("응답 생성 중..."):
                answer = ask_llm(user_text.strip(), images=images)
            st.markdown("#### 답변")
            st.markdown(answer)
            if tts_toggle:
                fn = speak_text(answer)
                if fn and os.path.exists(fn):
                    autoplay_audio_from_file(fn)

# 대화 기록
st.divider()
st.subheader("대화 기록")
if st.session_state.history:
    for role, text in st.session_state.history[-40:]:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)
else:
    st.caption("아직 대화가 없습니다.")
# ===== UI 끝 =====
