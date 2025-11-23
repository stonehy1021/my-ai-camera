import streamlit as st
import cv2
import mediapipe as mp
import time
import os
import numpy as np
from pathlib import Path

# ---------------- 1. 기본 설정 ----------------
st.set_page_config(page_title="AI 자동 촬영기", layout="wide")

# 윈도우 알림음 설정 (맥/리눅스에서는 에러 방지 위해 pass)
try:
    import winsound
except ImportError:
    winsound = None

# 저장 폴더 생성
SAVE_DIR = Path("captures")
SAVE_DIR.mkdir(exist_ok=True)

# ---------------- 2. 사이드바 설정 ----------------
st.sidebar.title("⚙️ 설정 패널")

st.sidebar.subheader("1. 각도 범위 (Z-Diff)")
# 요청하신 범위 (0.23 ~ 0.28)
min_val = st.sidebar.slider("최소 각도", 0.10, 0.40, 0.23, 0.01)
max_val = st.sidebar.slider("최대 각도", 0.10, 0.40, 0.28, 0.01)

st.sidebar.subheader("2. 카메라 선택")
# 0번이 내장, 1번이 연결된 폰(DroidCam)일 확률이 높습니다.
camera_id = st.sidebar.number_input("카메라 번호 (0 또는 1)", 0, 5, 0)

# ---------------- 3. Mediapipe 초기화 ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ---------------- 4. 메인 화면 ----------------
st.title("📸 AI 자동 촬영기 (Streamlit)")
st.markdown(f"""
### 🎯 목표 각도: **{min_val} ~ {max_val}**
**초록색 테두리**가 뜨면 **1.5초 뒤**에 소리와 함께 찍힙니다!
""")

# 실행 버튼
run = st.checkbox("🚀 카메라 켜기", value=False)

# 영상이 나올 공간 (빈 이미지로 자리 잡기)
frame_window = st.image([])
status_area = st.empty() # 상태 메시지용

# ---------------- 5. 실행 로직 ----------------
if run:
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        st.error(f"🚨 카메라({camera_id}번)를 열 수 없습니다. 사이드바에서 번호를 변경해보세요.")
    else:
        # 상태 변수들
        last_capture_time = 0
        enter_time = None
        capture_triggered = False
        flash_frames = 0 # 플래시 효과용

        while run and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("화면을 읽을 수 없습니다.")
                break

            # 1. 전처리
            frame = cv2.flip(frame, 1)  # 거울 모드
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. 얼굴 분석
            results = face_mesh.process(rgb_frame)

            current_z = 0.0
            in_range = False
            
            # 기본 디자인 (빨강)
            border_color = (0, 0, 255) 
            text_color = (0, 0, 255)
            status_text = "Adjust Angle"

            # 3. 플래시 효과 (촬영 직후 화면 하얗게)
            if flash_frames > 0:
                flash_frames -= 1
                white = np.full((h, w, 3), 255, dtype=np.uint8)
                frame = cv2.addWeighted(frame, 0.5, white, 0.5, 0)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Z-Diff 계산
                chin = landmarks[152].z
                forehead = landmarks[10].z
                current_z = chin - forehead

                # 범위 체크
                if min_val <= current_z <= max_val:
                    in_range = True
                    status_text = "HOLD ON!"
                    border_color = (0, 255, 0) # 초록
                    text_color = (0, 255, 0)

                # 4. 화면에 그리기 (UI 강화)
                # 테두리 그리기 (두껍게)
                cv2.rectangle(frame, (0, 0), (w, h), border_color, 20)

                # 현재 값 표시 (그림자 효과로 잘 보이게)
                info_text = f"Angle: {current_z:.4f}"
                cv2.putText(frame, info_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 6)
                cv2.putText(frame, info_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
                
                # 목표 범위 표시
                cv2.putText(frame, f"Target: {min_val}~{max_val}", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                # 5. 자동 촬영 로직
                if in_range:
                    if enter_time is None:
                        enter_time = time.time()

                    elapsed = time.time() - enter_time
                    countdown = 1.5 - elapsed

                    if countdown > 0:
                        # 카운트다운 숫자 표시 (화면 중앙)
                        cx, cy = w // 2, h // 2
                        cv2.putText(frame, f"{countdown:.1f}", (cx - 60, cy + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
                        cv2.putText(frame, f"{countdown:.1f}", (cx - 60, cy + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 4)
                    else:
                        # [촬영 시점]
                        if not capture_triggered:
                            if time.time() - last_capture_time > 3:
                                # 저장
                                ts = int(time.time())
                                filename = SAVE_DIR / f"AutoShot_{ts}.jpg"
                                # OpenCV 이미지는 BGR이므로 저장할 때는 그대로 둠 (Streamlit 표시는 RGB 변환해서 씀)
                                cv2.imwrite(str(filename), frame)

                                # 효과: 소리
                                if winsound:
                                    winsound.Beep(1500, 150) # 삑!
                                
                                # 효과: 알림 메시지
                                st.toast(f"📸 찰칵! 저장됨: {filename}", icon="✅")
                                
                                # 효과: 플래시 트리거
                                flash_frames = 5 
                                
                                last_capture_time = time.time()
                                capture_triggered = True
                else:
                    # 범위를 벗어나면 타이머 리셋
                    enter_time = None
                    capture_triggered = False
            else:
                # 얼굴 없음
                cv2.putText(frame, "No Face", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # 6. 화면 업데이트 (BGR -> RGB 변환하여 Streamlit에 표시)
            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_display)

            # CPU 부하 조절
            time.sleep(0.01)

    cap.release()


