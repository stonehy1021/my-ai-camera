import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import av
import numpy as np
import time
import queue

# ---------------- 1. 기본 설정 ----------------
st.set_page_config(page_title="mz구도 촬영기 (저장가능)", layout="centered")

# 세션 상태 초기화
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

st.title("📸 mz구도 자동 촬영기 ")
st.info("설정한 각도에 맞춰 고개를 들면 3초 뒤 자동 촬영됩니다!")

# ---------------- 2. 사이드바 설정 ----------------
st.sidebar.header("⚙️ 설정")
# 사용자가 설정한 값을 Processor로 넘겨야 함
min_val = st.sidebar.slider("최소 각도 (Z)", 0.0, 0.5, 0.13, 0.01)
max_val = st.sidebar.slider("최대 각도 (Z)", 0.0, 0.5, 0.25, 0.01)

# ---------------- 3. Mediapipe 초기화 ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- 4. 영상 처리 클래스 ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.enter_time = None
        self.capture_triggered = False
        self.last_capture_time = 0
        self.flash_frame = 0
        self.result_queue = queue.Queue()
        
        # 외부에서 설정값을 받을 변수 (기본값)
        self.min_val = 0.02
        self.max_val = 0.20

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        current_z = 0.0
        in_range = False
        border_color = (0, 0, 255) # 빨강
        status_msg = "Adjust Angle"
        
        # 플래시 효과 로직
        if self.flash_frame > 0:
            self.flash_frame -= 1
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            chin = landmarks[152].z
            forehead = landmarks[10].z
            
            # [수정됨] * -1 제거하여 양수 값 확보
            current_z = (chin - forehead)
            
            # [수정됨] 슬라이더 설정값 적용
            if self.min_val <= current_z <= self.max_val:
                in_range = True
                border_color = (0, 255, 0) # 초록
                status_msg = "HOLD ON!"
            
            # 화면 그리기
            cv2.rectangle(img, (0,0), (w,h), border_color, 20)
            
            # 디버깅용: 현재 Z값과 목표 범위 표시
            info_text = f"Z: {current_z:.3f} (Goal: {self.min_val}~{self.max_val})"
            cv2.putText(img, info_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)
            cv2.putText(img, status_msg, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            
            if in_range:
                if self.enter_time is None:
                    self.enter_time = time.time()
                
                elapsed = time.time() - self.enter_time
                countdown = 1.5 - elapsed
                
                if countdown > 0:
                    cx, cy = w//2, h//2
                    cv2.putText(img, f"{countdown:.1f}", (cx-50, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5)
                else:
                    # ★ 촬영 시점 ★
                    if not self.capture_triggered:
                        if time.time() - self.last_capture_time > 3:
                            save_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            self.result_queue.put(save_img)
                            self.last_capture_time = time.time()
                            self.capture_triggered = True
                            self.flash_frame = 5
            else:
                self.enter_time = None
                self.capture_triggered = False
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- 5. UI 로직 ----------------

# 1. 사진이 찍힌 상태 (결과 화면)
if st.session_state.snapshot is not None:
    st.success("📸 촬영 완료!")
    
    # 이미지 표시
    st.image(st.session_state.snapshot, caption="결과물", use_container_width=True)
    
    # 저장 버튼 생성
    img_bgr = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", img_bgr)
    
    if is_success:
        st.download_button(
            label="📥 사진 저장하기",
            data=buffer.tobytes(),
            file_name=f"MZ_Shot_{int(time.time())}.jpg",
            mime="image/jpeg",
            type="primary",
            use_container_width=True
        )
    
    # [변경됨] 다시 찍기 버튼 삭제하고 안내 문구 추가
    st.warning("🔄 다시 촬영하시려면 웹페이지를 새로고침 해주세요.")

# 2. 촬영 모드 (사진이 없는 상태)
else:
    # 카메라 실행
    ctx = webrtc_streamer(
        key="mz-camera",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    )

    # [중요] 실시간으로 슬라이더 값을 Processor 클래스에 주입
    if ctx.video_processor:
        ctx.video_processor.min_val = min_val
        ctx.video_processor.max_val = max_val

    # 사진 수신 대기 루프
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                try:
                    result_img = ctx.video_processor.result_queue.get(timeout=0.1)
                    if result_img is not None:
                        st.session_state.snapshot = result_img
                        st.rerun() # 화면 전환
                except queue.Empty:
                    pass
            time.sleep(0.1)
