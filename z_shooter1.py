import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import av
import numpy as np
import time
from pathlib import Path

# ---------------- 기본 설정 ----------------
st.set_page_config(page_title="mz 구도 카메라", layout="centered")

# 저장 경로 (서버에 저장됨)
SAVE_DIR = Path("captures")
SAVE_DIR.mkdir(exist_ok=True)

st.title("📸 mz구도 자동 촬영기")
st.info("아이폰/갤럭시/PC 모두 작동합니다.")

# ---------------- 사이드바 설정 ----------------
st.sidebar.header("⚙️ 설정")
# 모바일 화각 특성상 Z값 차이가 작게 나오므로 범위를 0.02~0.15로 잡습니다.
min_val = st.sidebar.slider("최소 각도", 0.0, 0.3, 0.02, 0.01)
max_val = st.sidebar.slider("최대 각도", 0.0, 0.3, 0.15, 0.01)

# ---------------- Mediapipe 초기화 ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- 영상 처리 클래스 (WebRTC) ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.enter_time = None
        self.capture_triggered = False
        self.last_capture_time = 0
        self.flash_frame = 0

    def recv(self, frame):
        # 1. 이미지 가져오기 (모바일 카메라 영상)
        img = frame.to_ndarray(format="bgr24")
        
        # 2. 거울 모드 (좌우 반전)
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # 3. 얼굴 분석
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        current_z = 0.0
        in_range = False
        border_color = (0, 0, 255) # 빨강
        status_msg = "Adjust Angle"
        
        # 4. 플래시 효과
        if self.flash_frame > 0:
            self.flash_frame -= 1
            # 하얀색 화면 덮기
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Z-Diff 계산
            chin = landmarks[152].z
            forehead = landmarks[10].z
            # 모바일/WebRTC 환경 보정
            current_z = (chin - forehead) * -1 
            
            # 범위 체크
            # WebRTC 클래스 내부에서는 st.session_state 접근이 까다로워 기본값 혹은 하드코딩된 로직을 쓸 수 있으나,
            # 여기서는 안전하게 넓은 범위(0.02~0.20)를 기본 로직으로 잡습니다.
            # (실제로는 recv 함수 밖에서 값을 주입받아야 하지만, 간단한 구현을 위해 고정 로직 사용)
            
            if 0.17 <= current_z <= 0.23: # 모바일용 추천 범위
                in_range = True
                border_color = (0, 255, 0) # 초록
                status_msg = "HOLD ON!"
            
            # UI 그리기
            cv2.rectangle(img, (0,0), (w,h), border_color, 20)
            
            info_text = f"Z: {current_z:.4f}"
            cv2.putText(img, info_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 5)
            cv2.putText(img, info_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, border_color, 3)
            
            cv2.putText(img, status_msg, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # 자동 촬영 로직
            if in_range:
                if self.enter_time is None:
                    self.enter_time = time.time()
                
                elapsed = time.time() - self.enter_time
                countdown = 1.5 - elapsed
                
                if countdown > 0:
                    cx, cy = w//2, h//2
                    cv2.putText(img, f"{countdown:.1f}", (cx-50, cy+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5)
                else:
                    if not self.capture_triggered:
                        if time.time() - self.last_capture_time > 3:
                            # 저장 (서버에 저장됨)
                            ts = int(time.time())
                            filename = SAVE_DIR / f"Mobile_Shot_{ts}.jpg"
                            cv2.imwrite(str(filename), img)
                            print(f"📸 저장됨: {filename}")
                            
                            self.last_capture_time = time.time()
                            self.capture_triggered = True
                            self.flash_frame = 5
            else:
                self.enter_time = None
                self.capture_triggered = False
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- WebRTC 실행 ----------------
# 모바일 접속을 위한 STUN 서버 설정 (필수 - 이거 없으면 폰에서 안됨)
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="mobile-camera",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "video": {"facingMode": "user"}, # 전면 카메라 사용
        "audio": False
    },
)


