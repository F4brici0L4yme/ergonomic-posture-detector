import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calcular_angulo(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cosangulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    ang = np.arccos(np.clip(cosangulo, -1.0, 1.0))
    return np.degrees(ang)

def coord(landmark, w, h):
    return (landmark.x * w, landmark.y * h)

def main():
    cap = cv2.VideoCapture(0)

    postura = "NO DETECTADO"
    postura_pierna = "NO DETECTADO"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            LEFT_EAR      = mp_pose.PoseLandmark.LEFT_EAR.value
            LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            LEFT_HIP      = mp_pose.PoseLandmark.LEFT_HIP.value

            ear      = coord(lm[LEFT_EAR], w, h)
            shoulder = coord(lm[LEFT_SHOULDER], w, h)
            hip      = coord(lm[LEFT_HIP], w, h)

            angulo_dorso = calcular_angulo(ear, shoulder, hip)

            cv2.circle(frame, (int(ear[0]), int(ear[1])), 5, (0,255,255), -1)
            cv2.circle(frame, (int(shoulder[0]), int(shoulder[1])), 5, (0,255,0), -1)
            cv2.circle(frame, (int(hip[0]), int(hip[1])), 5, (255,0,0), -1)

            cv2.line(frame, (int(ear[0]), int(ear[1])),
                            (int(shoulder[0]), int(shoulder[1])), (0,255,0), 2)

            cv2.line(frame, (int(shoulder[0]), int(shoulder[1])),
                            (int(hip[0]), int(hip[1])), (0,255,0), 2)

            cv2.putText(frame, f"Angulo dorso: {angulo_dorso:.1f}°",
                        (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            if 160 <= angulo_dorso <= 175:
                postura = "POSTURA CORRECTA"
            else:
                postura = "MALA POSTURA"

            LEFT_KNEE  = mp_pose.PoseLandmark.LEFT_KNEE.value
            LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value

            knee  = coord(lm[LEFT_KNEE], w, h)
            ankle = coord(lm[LEFT_ANKLE], w, h)

            angulo_pierna = calcular_angulo(hip, knee, ankle)

            cv2.circle(frame, (int(knee[0]), int(knee[1])), 5, (255,0,255), -1)
            cv2.circle(frame, (int(ankle[0]), int(ankle[1])), 5, (0,255,255), -1)

            cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255,255,0), 2)
            cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (255,255,0), 2)

            cv2.putText(frame, f"Angulo pierna: {angulo_pierna:.1f}°",
                        (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            if 80 <= angulo_pierna <= 100:
                postura_pierna = "PIERNA CORRECTA"
            else:
                postura_pierna = "PIERNA INCORRECTA"

        cv2.putText(
            frame, postura, (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0,255,0) if postura == "POSTURA CORRECTA" else (0,0,255),
            3
        )

        cv2.putText(
            frame, postura_pierna, (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0,255,0) if postura_pierna == "PIERNA CORRECTA" else (0,0,255),
            3
        )

        cv2.imshow("Deteccion de Dorso (Lado Izquierdo)", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
