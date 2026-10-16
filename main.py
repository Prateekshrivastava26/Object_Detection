from ultralytics import YOLO
import cv2
import time
import os
from datetime import datetime
from audio.vad_detector import VADDetector
import mediapipe as mp

print("Starting PRO Cheating Detection System...")

print("1 → Laptop")
print("2 → Mobile / Earphone")
print("3 → Cheating Detection")
print("4 → Head Movement Detection")
print("5 → ALL")

choice = input("Enter choice: ")

# -------- LOAD MODELS --------
model1 = None
model2 = None
model3 = None

if choice == "1":
    model1 = YOLO("runs/detect/train/weights/best.pt")
    mode = 1

elif choice == "2":
    model2 = YOLO("runs/detect/train2/weights/best.pt")
    mode = 2

elif choice == "3":
    model2 = YOLO("runs/detect/train2/weights/best.pt")
    model3 = YOLO("runs/detect/train3/weights/best.pt")
    mode = 3

elif choice == "4":
    mode = 4

elif choice == "5":
    model1 = YOLO("runs/detect/train/weights/best.pt")
    model2 = YOLO("runs/detect/train2/weights/best.pt")
    model3 = YOLO("runs/detect/train3/weights/best.pt")
    mode = 5

else:
    print("Invalid choice")
    exit()

# -------- MEDIAPIPE --------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=50, refine_landmarks=True)

# -------- CONFIG --------
DEVICES = {
    "phone": (0, 0, 255),
    "earphone": (255, 0, 0),
    "bluetooth": (0, 165, 255),
    "laptop": (0, 205, 255),
    "person": (0, 155, 255),
    "cheating": (0, 0, 255),

    "head_left": (255, 0, 255),
    "head_right": (0, 255, 255),
    "head_down": (255, 255, 0),
    "head_up": (0, 128, 255),
    "head_front": (0, 255, 0),
}

CONF = 0.25
SCREENSHOT_DELAY = 3
VOICE_DELAY = 5

last_screenshot_time = {}

cap = cv2.VideoCapture(0)

vad = VADDetector()
vad.start()

voice_start_time = None

os.makedirs("Cheating_Detected", exist_ok=True)

# -------- YOLO FUNCTION --------
def process_model(model, frame, detected_events, labels_map=None, only_devices=False):
    results = model(frame, conf=CONF)[0]

    for box in results.boxes:
        label = model.names[int(box.cls)].lower().strip()
        conf_val = float(box.conf)

        if label in ["mobile", "cell phone"]:
            label = "phone"
        elif label in ["earphones"]:
            label = "earphone"

        if labels_map and label in labels_map:
            label = labels_map[label]

        if only_devices and label not in ["phone", "earphone", "bluetooth"]:
            continue

        if label not in DEVICES or conf_val < 0.2:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detected_events.append(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), DEVICES[label], 3)
        cv2.putText(frame, f"{label} {conf_val:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DEVICES[label], 2)

# -------- HEAD FUNCTION (FINAL FIXED) --------
def detect_head(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
     
    if not results.multi_face_landmarks:
        return []

    h, w, _ = frame.shape
    detections = []
     
    for face in results.multi_face_landmarks:
        xs = [int(p.x * w) for p in face.landmark]
        ys = [int(p.y * h) for p in face.landmark]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        nose = face.landmark[1]
        chin = face.landmark[152]
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]
        left_cheek = face.landmark[234]
        right_cheek = face.landmark[454]

        nose_x = nose.x * w
        nose_y = nose.y * h
        chin_y = chin.y * h

        eye_avg_y = ((left_eye.y + right_eye.y) / 2) * h

        # UP/DOWN
        ratio = (nose_y - eye_avg_y) / (chin_y - eye_avg_y + 1e-6)

        # LEFT/RIGHT (stable)
        left_x = left_cheek.x * w
        right_x = right_cheek.x * w
        face_center = (left_x + right_x) / 2

        horizontal_shift = (nose_x - face_center) / (right_x - left_x + 1e-6)

        label = "head_front"

        if ratio > 0.58:
            label = "head_down"
        elif ratio < 0.18:
            label = "head_up"
        else:
            if horizontal_shift < -0.50:
                label = "head_left"
            elif horizontal_shift > 0.40:
                label = "head_right"

        detections.append(((x1, y1, x2, y2), label))

    return detections

# -------- MAIN LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected_events = []
    head_alert = False

    # -------- VOICE --------
    if vad.voice_detected:
        if voice_start_time is None:
            voice_start_time = time.time()

        if time.time() - voice_start_time >= VOICE_DELAY:
            cv2.putText(frame, "VOICE ALERT", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            detected_events.append("voice")
    else:
        voice_start_time = None

    # -------- YOLO --------
    if mode == 1:
        process_model(model1, frame, detected_events)

    elif mode == 2:
        process_model(model2, frame, detected_events, only_devices=True)
    
    elif mode == 3:
        process_model(model2, frame, detected_events, only_devices=True)
        process_model(model3, frame, detected_events,
                      {"students_cheating": "cheating"})

    elif mode == 5:
        process_model(model1, frame, detected_events)
        process_model(model2, frame, detected_events, only_devices=True)
        process_model(model3, frame, detected_events,
                      {"students_cheating": "cheating"})

    # -------- HEAD --------
    if mode in [4, 5]:
        head_results = detect_head(frame)

        for i, (box, label) in enumerate(head_results):
            x1, y1, x2, y2 = box

            cv2.rectangle(frame, (x1, y1), (x2, y2), DEVICES[label], 3)
            cv2.putText(frame, f"{label}_{i+1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, DEVICES[label], 2)

            if label != "head_front":
                detected_events.append(f"{label}_{i}")
                head_alert = True

    # -------- SCREENSHOT --------
    current_time = time.time()

    for event in detected_events:
        delay = VOICE_DELAY if event == "voice" else SCREENSHOT_DELAY

        if event not in last_screenshot_time:
            last_screenshot_time[event] = current_time
            continue

        if current_time - last_screenshot_time[event] >= delay:
            now = datetime.now()

            frame_copy = frame.copy()
            cv2.putText(frame_copy, f"{event.upper()} | {now}",
                        (20, frame_copy.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            filename = f"Cheating_Detected/{event}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame_copy)

            print(f"{event} screenshot saved")
            last_screenshot_time[event] = current_time

    # cleanup
    for event in list(last_screenshot_time.keys()):
        if event not in detected_events:
            del last_screenshot_time[event]

    # -------- ALERT UI --------
    if head_alert:
        cv2.putText(frame, "HEAD ALERT", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("PRO Cheating Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()