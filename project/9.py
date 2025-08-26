# =============================================================
# Streamlit 반도체 공정 Q&A (음성 + 이미지 여러장 + 카메라 + 채팅 + 자동 STT/TTS)
# 파일명: semifab_voice_image_chat_app.py
# -------------------------------------------------------------
# 의존: streamlit, audio_recorder_streamlit, pillow, faster_whisper, langchain-openai
# -------------------------------------------------------------

import os
import io
import base64
from typing import List, Tuple, Optional

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from PIL import Image

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# STT
from faster_whisper import WhisperModel

# TTS (선택)
try:
    import openai  # type: ignore
except Exception:  # 라이브러리 미설치 시 무음 처리
    openai = None

APP_TITLE = "반도체 공정 Q&A"

SYSTEM_PROMPT = (
    "당신은 반도체 공정(포토, 식각, 증착, CMP, 이온주입, 확산, 열처리 등) 멘토입니다.\n"
    "규칙:\n"
    "- 먼저 의도: 를 한 줄로 제시\n"
    "- 복잡한 주제는 단계 1, 2, ... 형태로 설명\n"
    "- 정보 부족 시 '정보가 부족합니다' 명시\n"
    "- 추론 필요 시 '추론(유형: 연역/귀납/유추): 근거' 1줄 추가\n"
    "- 항상 정중한 한국어(존댓말)로 답변\n"
)

# -------------------------------------------------------------
# 세션 초기화
# -------------------------------------------------------------

def _init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history: List[Tuple[str, str]] = []
    if "upload_images" not in st.session_state:
        st.session_state.upload_images: List[Image.Image] = []  # 다중 업로드 이미지
    if "camera_images" not in st.session_state:
        st.session_state.camera_images: List[Image.Image] = []  # 다중 카메라 캡처
    if "use_upload_for_next" not in st.session_state:
        st.session_state.use_upload_for_next = False
    if "use_camera_for_next" not in st.session_state:
        st.session_state.use_camera_for_next = False

# -------------------------------------------------------------
# 유틸: 이미지 → data URL
# -------------------------------------------------------------

def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

# -------------------------------------------------------------
# STT
# -------------------------------------------------------------

def load_whisper(model_size: str = "base") -> WhisperModel:
    return WhisperModel(model_size, device="auto", compute_type="int8")

def transcribe_audio_bytes(audio_bytes: bytes, model_size: str = "base") -> str:
    if not audio_bytes:
        return ""
    tmp_path = "_tmp_query.wav"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    model = load_whisper(model_size)
    segments, _ = model.transcribe(tmp_path, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", ""))
    return text.strip()

# -------------------------------------------------------------
# LLM
# -------------------------------------------------------------

def get_llm(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=0.2)

def ask_llm(query_text: str, images: Optional[List[Image.Image]] = None) -> str:
    """텍스트 + (선택) 다중 이미지로 질의."""
    sys_msgs = [SystemMessage(content=SYSTEM_PROMPT)]

    # 사용자 메시지 구성
    if images:
        content: List[dict] = [{"type": "text", "text": query_text}]
        for img in images:
            data_url = pil_to_data_url(img, fmt="PNG")
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        user_msg = HumanMessage(content=content)
    else:
        user_msg = HumanMessage(content=query_text)

    llm = get_llm()
    response = llm.invoke(sys_msgs + [user_msg])

    # 로그
    st.session_state.history.append(("user", query_text))
    st.session_state.history.append(("assistant", response.content))

    return response.content

# -------------------------------------------------------------
# TTS (선택) + 자동 재생 헬퍼
# -------------------------------------------------------------

def speak_text(text: str, filename: str = "tts_output.mp3") -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or openai is None:
        return None
    try:
        client = openai.OpenAI(api_key=api_key)  # 최신 SDK 기준
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",  # 환경에 따라 tts-1 계열 사용 가능
            voice="alloy",
            input=text,
        )
        with open(filename, "wb") as f:
            f.write(resp.read())
        return filename
    except Exception:
        return None

# 오디오 자동 재생(브라우저) — Streamlit audio 대신 HTML5 audio autoplay 사용
import streamlit.components.v1 as components
import base64 as _b64

def autoplay_audio_from_file(filepath: str) -> None:
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        b64 = _b64.b64encode(data).decode("utf-8")
        html = f"""
        <audio autoplay playsinline>
            <source src='data:audio/mpeg;base64,{b64}' type='audio/mpeg'>
        </audio>
        """
        components.html(html, height=0)
    except Exception:
        pass
    try:
        client = openai.OpenAI(api_key=api_key)  # 최신 SDK 기준
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",  # 환경에 따라 tts-1 계열 사용 가능
            voice="alloy",
            input=text,
        )
        with open(filename, "wb") as f:
            f.write(resp.read())
        return filename
    except Exception:
        return None

# -------------------------------------------------------------
# 헬퍼: 다음 질문에 포함할 이미지 셀렉션(다중)
# -------------------------------------------------------------

def get_selected_images() -> List[Image.Image]:
    """체크 상태에 따라 다중 이미지 리스트 반환. 우선순위: 카메라 체크 시 카메라 이미지들 먼저."""
    selected: List[Image.Image] = []
    if st.session_state.use_camera_for_next:
        selected.extend(st.session_state.camera_images)
    if st.session_state.use_upload_for_next:
        selected.extend(st.session_state.upload_images)
    return selected

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------

def render_chat_boxes() -> None:
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
        files = st.file_uploader("공정 사진/도표 여러 장 업로드", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
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
            st.image(st.session_state.camera_images, caption=[f"카메라 {i+1}" for i in range(len(st.session_state.camera_images))])
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
        col_s1, col_s2 = st.columns([1,1])
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

# -------------------------------------------------------------
# 메인
# -------------------------------------------------------------

def main() -> None:
    _init_session_state()
    render_chat_boxes()
    st.divider()
    st.subheader("대화 기록")
    if st.session_state.history:
        for role, text in st.session_state.history[-40:]:
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(text)
    else:
        st.caption("아직 대화가 없습니다.")

if __name__ == "__main__":
    main()
