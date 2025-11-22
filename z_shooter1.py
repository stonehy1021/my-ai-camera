import streamlit as st
import cv2
import mediapipe as mp
import time
import os
import numpy as np

# ---------------- 기본 설정 ----------------
st.set_page_config(page_title="AI 자동 촬영기", layout="wide")

# 윈도우 알림음 설정 (맥/리눅스 에러 방지)
try:
    import winsound
except ImportError:
    winsound = None

# 저장 폴더 생성
SAVE_DIR = "captures"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ---------------- 사이드바 설정 ----------------
st.sidebar.title("⚙️ 설정 패널")

# [요청하신 각도 범위 고정]
# 슬라이더로 조절 가능하지만 기본값을 0.23 ~ 0.28로 설정했습니다.
st.sidebar.subheader("1. 각도 범위 (Z-Diff)")
min_val = st.sidebar.slider("최소 각도", 0.10, 0.40, 0.23, 0.01)
max_val = st.sidebar.slider("최대 각도", 0.10, 0.40, 0.28, 0.01)

st.sidebar.subheader("2. 카메라 선택")
camera_id = st.sidebar.number_input("카메라 번호 (0:기본, 1:외부)", 0, 5, 0)

# ---------------- Mediapipe 초기화 ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ---------------- 메인 화면 ----------------
st.title("📸 AI 자동 촬영기 (Local)")
st.markdown(f"""
### 🎯 목표 각도: **{min_val} ~ {max_val}**
카메라를 켜고 고개를 움직여 **Z-Diff** 수치를 맞춰보세요.  
초록색 숫자가 뜨면 **1.5초 뒤에 자동으로 찍힙니다!**
""")

# 실행 버튼
run = st.checkbox("🚀 카메라 켜기 (체크하면 시작)", value=False)

# 영상이 나올 공간
frame_window = st.image([])

# ---------------- 실행 로직 ----------------
if run:
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        st.error(f"🚨 카메라({camera_id}번)를 열 수 없습니다. 번호를 1로 바꿔보거나 다른 프로그램(Zoom 등)을 꺼주세요.")
    else:
        # 상태 변수
        last_capture_time = 0
        enter_time = None
        capture_triggered = False

        while run and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("화면을 읽을 수 없습니다.")
                break

            # 1. 이미지 전처리
            frame = cv2.flip(frame, 1)  # 거울 모드
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. 얼굴 분석
            results = face_mesh.process(rgb_frame)

            current_z = 0.0
            in_range = False
            status_text = "Out of Range"
            color = (0, 0, 255)  # 빨강 (불일치)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Z-Diff 계산
                chin = landmarks[152].z
                forehead = landmarks[10].z
                current_z = chin - forehead

                # 범위 체크
                if min_val <= current_z <= max_val:
                    in_range = True
                    status_text = "Target Locked!"
                    color = (0, 255, 0)  # 초록 (일치)

                # 3. 화면에 정보 표시 (크고 잘 보이게)
                # 현재 값
                cv2.putText(frame, f"Current: {current_z:.4f}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                # 목표 범위
                cv2.putText(frame, f"Target: {min_val} ~ {max_val}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                # 4. 자동 촬영 로직
                if in_range:
                    if enter_time is None:
                        enter_time = time.time()

                    # 경과 시간 계산
                    elapsed = time.time() - enter_time

                    # 1.5초 카운트다운 (기존 3초는 너무 길어서 줄임)
                    countdown = 1.5 - elapsed

                    if countdown > 0:
                        # 카운트다운 표시
                        center_x, center_y = w // 2, h // 2
                        cv2.circle(frame, (center_x, center_y), 100, (0, 255, 255), 5)
                        cv2.putText(frame, f"{countdown:.1f}", (center_x - 40, center_y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
                    else:
                        # [촬영 시점]
                        if not capture_triggered:
                            # 연속 촬영 방지 쿨타임 (3초)
                            if time.time() - last_capture_time > 3:
                                # 저장
                                ts = int(time.time())
                                filename = f"{SAVE_DIR}/AutoShot_{ts}.jpg"
                                cv2.imwrite(filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

                                # 효과 (소리 + 화면 번쩍)
                                if winsound: winsound.Beep(1000, 150)
                                cv2.rectangle(frame, (0, 0), (w, h), (255, 255, 255), -1)  # 전체 흰색 화면(플래시)

                                st.toast(f"📸 찰칵! 저장됨: {filename}")
                                last_capture_time = time.time()
                                capture_triggered = True
                else:
                    # 범위를 벗어나면 타이머 리셋
                    enter_time = None
                    capture_triggered = False
            else:
                cv2.putText(frame, "No Face", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 5. 화면 업데이트
            frame_window.image(frame, channels="BGR")

            # CPU 점유율 낮추기 (부드러운 실행)
            time.sleep(0.03)

    cap.release()