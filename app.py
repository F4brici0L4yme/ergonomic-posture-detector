import cv2
import torch
import yolov5
import mediapipe as mp
import math
import time
import threading
from playsound import playsound

MODEL_PATH = 'small640.pt'
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50

def calcular_angulo(a, b, c):
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot / (mag_ba * mag_bc + 1e-9)
    return math.degrees(math.acos(max(min(cos_angle, 1), -1)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = yolov5.load(MODEL_PATH, device=device)
model.conf = CONFIDENCE_THRESHOLD
model.iou = IOU_THRESHOLD
model.max_det = 1

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

bad_back_start = None
bad_leg_start = None
alert_back_active = False
alert_leg_active = False
last_back_alert = 0
last_leg_alert = 0
cooldown = 5

def play_back():
    playsound("dorso.mp3")

def play_leg():
    playsound("piernas.mp3")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    rendered_frame = results.render()[0].copy()

    det = results.xyxy[0]
    posture = None
    if len(det) > 0:
        _, _, _, _, conf, class_id = det[0]
        posture = "bad" if int(class_id) == 1 else "good"

    now = time.time()

    if posture == "bad":
        if bad_back_start is None:
            bad_back_start = now
        if (now - bad_back_start >= 5) and not alert_back_active:
            threading.Thread(target=play_back).start()
            alert_back_active = True
            last_back_alert = now
        if alert_back_active and (now - last_back_alert >= cooldown):
            threading.Thread(target=play_back).start()
            last_back_alert = now
        cv2.putText(rendered_frame, "ESPALDA INCORRECTA", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        bad_back_start = None
        alert_back_active = False
        cv2.putText(rendered_frame, "ESPALDA CORRECTA", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = pose.process(rgb)

    if mp_results.pose_landmarks:
        lm = mp_results.pose_landmarks.landmark
        h, w, _ = frame.shape
        A = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
             int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        B = (int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w),
             int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
        C = (int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w),
             int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))

        angulo = calcular_angulo(A, B, C)

        cv2.circle(rendered_frame, A, 6, (0,255,0), -1)
        cv2.circle(rendered_frame, B, 6, (0,255,0), -1)
        cv2.circle(rendered_frame, C, 6, (0,255,0), -1)
        cv2.line(rendered_frame, A, B, (0,255,0), 2)
        cv2.line(rendered_frame, B, C, (0,255,0), 2)
        cv2.putText(rendered_frame, f"Angulo: {int(angulo)} grados",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        leg_bad = not (80 <= angulo <= 100)

        if leg_bad:
            if bad_leg_start is None:
                bad_leg_start = now
            if (now - bad_leg_start >= 5) and not alert_leg_active:
                threading.Thread(target=play_leg).start()
                alert_leg_active = True
                last_leg_alert = now
            if alert_leg_active and (now - last_leg_alert >= cooldown):
                threading.Thread(target=play_leg).start()
                last_leg_alert = now
            cv2.putText(rendered_frame, "PIERNAS INCORRECTAS",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            bad_leg_start = None
            alert_leg_active = False
            cv2.putText(rendered_frame, "PIERNAS CORRECTAS",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('Detector de Postura', rendered_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
