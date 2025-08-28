import os
import io
import hashlib
from typing import List
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from PIL import Image

# LLM 모듈에서 필요한 것만 씀
from LLM import (
    APP_TITLE,
    init_session_state,
    transcribe_audio_bytes,
    ask_llm,
    speak_text,
    autoplay_audio_from_file,
)

# 세션 초기화함
init_session_state()

# 세션 키 보강함
ss = st.session_state
ss.setdefault("upload_images", [])           # 업로드 이미지 리스트 담음
ss.setdefault("camera_images", [])           # 카메라 이미지 리스트 담음
ss.setdefault("use_upload_for_next", False)  # 업로드 이미지 사용 여부
ss.setdefault("use_camera_for_next", False)  # 카메라 이미지 사용 여부
ss.setdefault("last_camera_digest", None)    # 마지막 반영된 카메라 프레임 해시 저장함
ss.setdefault("chat_dialog", [])             # 채팅 로그 저장함
ss.setdefault("cam_key", 0)                  # camera_input 리셋용 키

st.title(APP_TITLE)

# 1) 음성 인식 (자동 질문/응답)
with st.container():
    st.markdown("### 1) 음성 인식 (자동 질문/응답)")
    st.caption("말하고 멈추면 텍스트 변환 → 답변 생성 → 선택 시 음성 자동 재생함")

    # 음성 녹음 버튼은 여기만 둠
    audio_bytes = audio_recorder(text="말하기")
    tts_auto = st.checkbox("음성 답변 자동 재생", value=True)

    # 음성 수신되면 처리함
    if audio_bytes:
        text_in = transcribe_audio_bytes(audio_bytes)
        if text_in:
            st.success(f"인식된 질문: {text_in}")

            # 이미지 선택 합침 (카메라 우선 → 업로드)
            images: List[Image.Image] = []
            if ss.get("use_camera_for_next", False):
                images.extend(ss.get("camera_images", []))
            if ss.get("use_upload_for_next", False):
                images.extend(ss.get("upload_images", []))

            with st.spinner("응답 생성 중..."):
                answer = ask_llm(text_in, images=images)

            st.markdown("#### 답변")
            st.markdown(answer)

            # 자동 음성 재생 옵션 처리함
            if tts_auto:
                fn = speak_text(answer)
                if fn and os.path.exists(fn):
                    autoplay_audio_from_file(fn)

# 2) 이미지 업로드 (여러 장)
with st.container():
    st.markdown("### 2) 이미지 업로드")
    files = st.file_uploader(
        "공정 사진/도표 여러 장 업로드",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    # 업로드 파일 PIL 변환 후 저장함
    if files:
        imgs: List[Image.Image] = []
        for f in files:
            try:
                img = Image.open(f).convert("RGB")
                imgs.append(img)
            except Exception:
                pass
        if imgs:
            ss.upload_images = imgs
            st.image(imgs, caption=[f"업로드 {i+1}" for i in range(len(imgs))])

    # 업로드 이미지 사용 여부 제공함
    col_u1, _ = st.columns([1, 1])
    with col_u1:
        ss.use_upload_for_next = st.checkbox(
            "이번 질문에 업로드 이미지 포함",
            value=ss.get("use_upload_for_next", False)
        )

# 3) 카메라 촬영 (여러 장)
with st.container():
    st.markdown("### 3) 카메라 촬영")
    # cam_key로 위젯 리셋 가능함 → 클리어 포토 효과
    cam = st.camera_input("카메라로 촬영", key=f"cam_input_{ss.cam_key}")

    # 카메라 입력 처리함
    # 중복 프레임 방지 위해 해시 사용함
    if cam is not None:
        raw = cam.getvalue()                       # 현재 프레임 바이트 얻음
        d = hashlib.md5(raw).hexdigest()           # 프레임 해시 계산함
        if ss.last_camera_digest != d:
            try:
                imgc = Image.open(io.BytesIO(raw)).convert("RGB")
                ss.camera_images.append(imgc)
                ss.last_camera_digest = d
                st.success("카메라 이미지 추가됨")
            except Exception:
                st.warning("카메라 이미지 로드 실패")

    # 현재 저장된 카메라 이미지 미리보기
    if ss.camera_images:
        st.image(
            ss.camera_images,
            caption=[f"카메라 {i+1}" for i in range(len(ss.camera_images))]
        )

    # 카메라 이미지 사용 여부 + 지우기 버튼 제공함
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        ss.use_camera_for_next = st.checkbox(
            "이번 질문에 카메라 이미지 포함",
            value=ss.get("use_camera_for_next", False)
        )
    with col_c2:
        # 지우기 버튼 누르면
        # 1) 세션에 저장된 카메라 이미지 비움
        # 2) 마지막 프레임 해시 초기화함
        # 3) cam_key 증가시켜 camera_input 위젯 재생성 → 클리어 포토 효과 발생
        if st.button("카메라 이미지 지우기"):
            ss.camera_images = []
            ss.last_camera_digest = None
            ss.cam_key += 1
            st.success("카메라 이미지 목록 비움 및 카메라 초기화 완료")
            st.rerun()

    st.caption("두 체크 모두 켜면 카메라 이미지 우선, 이후 업로드 이미지 순으로 전송됨")

# 4) 챗봇
st.header("질의응답")

# 상단 툴바: 대화 초기화, 음성 응답 옵션 제공함
col_a, col_b, col_c = st.columns([1, 1, 8])
with col_a:
    if st.button("대화 초기화", key="btn_clear_chat", use_container_width=True):
        ss.chat_dialog = []
        if "history" in ss:
            ss.history = []
        st.toast("대화 초기화 완료")
        (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()
with col_b:
    tts_chat = st.checkbox("챗봇 음성 응답", value=True)

# 과거 대화 뿌림
if ss.chat_dialog:
    for msg in ss.chat_dialog:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

# 입력 폼(텍스트만 사용). 챗봇 영역 녹음 버튼 없음
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input("질문을 입력하세요… (예: EUV와 DUV 차이)", key="chat_text")
    submitted = st.form_submit_button("Send")

# 폼 제출 처리함
if submitted:
    # 공백이면 안내 후 종료함
    if not user_text or not user_text.strip():
        st.info("질문을 입력해 주십시오.")
    else:
        # 사용자 메시지 기록함
        ss.chat_dialog.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # 최근 6턴만 맥락으로 합침 (user/assistant 합 12개)
        tail = ss.chat_dialog[-12:]
        hist_lines = []
        for m in tail:
            role = "사용자" if m["role"] == "user" else "도우미"
            hist_lines.append(f"{role}: {m['content']}")
        hist_txt = "\n".join(hist_lines)

        # 이미지 선택 합침 (카메라 우선 → 업로드)
        images: List[Image.Image] = []
        if ss.get("use_camera_for_next", False):
            images.extend(ss.get("camera_images", []))
        if ss.get("use_upload_for_next", False):
            images.extend(ss.get("upload_images", []))

        # 프롬프트 구성함
        full_prompt = (
            "아래의 최근 대화를 참고하여, 이어지는 사용자의 질문에 정중한 한국어(존댓말)로 답변 바람.\n\n"
            f"[최근 대화]\n{hist_txt or '(이전 대화 없음)'}\n\n"
            f"[질문]\n{user_text}"
        )

        # 답변 생성함
        with st.chat_message("assistant"):
            with st.spinner("응답 생성 중..."):
                answer = ask_llm(full_prompt, images=images)
            st.markdown(answer)

            # 어시스턴트 메시지 기록함
            ss.chat_dialog.append({"role": "assistant", "content": answer})

            # 음성 응답 옵션 처리함
            if tts_chat:
                fn = speak_text(answer)
                if fn and os.path.exists(fn):
                    autoplay_audio_from_file(fn)
