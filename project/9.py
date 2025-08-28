import os
from typing import List
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from PIL import Image
from LLM import get_llm_backend, build_vectorstore_from_pdfs, message
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

# 챗봇
st.markdown("---")
st.header("질의응답")

if "chat_dialog" not in st.session_state:
    st.session_state.chat_dialog = []   # [{'role':'user'|'assistant','content':str}]

# 상단 툴바: 대화 초기화 + 음성 재생 체크박스
col_a, col_b, col_c = st.columns([1, 1, 8])
with col_a:
    if st.button("대화 초기화", key="btn_clear_chat", use_container_width=True):
        st.session_state.chat_dialog = []
        # ask_llm 내부 history를 같이 쓰신다면 아래도 함께 비우세요 (선택)
        if "history" in st.session_state:
            st.session_state.history = []
        st.toast("대화가 초기화되었습니다. 🧹")
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()
with col_b:
    tts_chat = st.checkbox("챗봇 음성 응답", value=True)

# 과거 대화 렌더 (최근이 아래로 내려오도록 순서대로 출력)
if st.session_state.chat_dialog:
    for msg in st.session_state.chat_dialog:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

# 입력 폼(텍스트 + 선택적 음성 입력)
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input("질문을 입력하세요… (예: EUV와 DUV 차이)", key="chat_text")
    use_mic   = st.checkbox("음성 입력 사용", value=False)
    audio_chat_bytes = audio_recorder(text="녹음", key="chat_mic") if use_mic else None
    submitted = st.form_submit_button("Send")

# 제출 처리
if submitted:
    # 1) 음성 입력이 있고 텍스트가 비어있으면 음성부터 문자로 변환
    if (not user_text or not user_text.strip()) and audio_chat_bytes:
        try:
            user_text = transcribe_audio_bytes(audio_chat_bytes) or ""
        except Exception:
            user_text = ""

    if not user_text or not user_text.strip():
        st.info("질문을 입력하거나 음성으로 말씀해 주세요.")
    else:
        # 2) 사용자 메시지 기록 및 표시
        st.session_state.chat_dialog.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # 3) 최근 대화 맥락(간단) 구성
        #   - 마지막 6개 메시지를 '사용자/도우미:' 형태로 합쳐 LLM에 컨텍스트로 전달
        tail = st.session_state.chat_dialog[-12:]  # user/assistant 합쳐 최대 12개 = 최근 6턴
        hist_lines = []
        for m in tail:
            role = "사용자" if m["role"] == "user" else "도우미"
            hist_lines.append(f"{role}: {m['content']}")
        hist_txt = "\n".join(hist_lines)

        # 4) 이미지 선택(업로드/카메라) 포함
        images = get_selected_images()

        # 5) 프롬프트 구성 → ask_llm 호출
        #    (LLM.py의 SYSTEM_PROMPT가 적용되며, 여기서는 대화 맥락을 텍스트로 주입)
        full_prompt = (
            "아래의 최근 대화를 참고하여, 이어지는 사용자의 질문에 정중한 한국어(존댓말)로 답변해 주세요.\n\n"
            f"[최근 대화]\n{hist_txt or '(이전 대화 없음)'}\n\n"
            f"[질문]\n{user_text}"
        )

        with st.chat_message("assistant"):
            with st.spinner("응답 생성 중..."):
                answer = ask_llm(full_prompt, images=images)

            st.markdown(answer)
            # 6) 응답 저장
            st.session_state.chat_dialog.append({"role": "assistant", "content": answer})

            # 7) 음성 응답(선택)
            if tts_chat:
                fn = speak_text(answer)
                if fn and os.path.exists(fn):
                    autoplay_audio_from_file(fn)

