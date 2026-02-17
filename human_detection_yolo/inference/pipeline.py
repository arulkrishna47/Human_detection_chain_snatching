import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from collections import deque

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# YOLO SETTINGS (DISPLAY ONLY)
# =====================================================
YOLO_CONF = 0.45
YOLO_IOU = 0.30
IMG_SIZE = 640

yolo = YOLO("models/yolo/human_best.pt")

# =====================================================
# CHAIN SNATCH CNN (FULL FRAME)
# =====================================================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN().to(device)
model.load_state_dict(
    torch.load(
        "models/chain_snatch/epoch_5.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

# =====================================================
# TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =====================================================
# VIDEO
# =====================================================
cap = cv2.VideoCapture("videos_test/test3.avi")
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

# =====================================================
# EVENT PARAMETERS
# =====================================================
SNATCH_THRESHOLD = 0.6
CONSECUTIVE_FRAMES = 5
SMOOTHING_WINDOW = 5

prob_history = deque(maxlen=SMOOTHING_WINDOW)
consecutive = 0
snatch_detected = False

# =====================================================
# FUNCTIONS
# =====================================================
def cnn_predict(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(image)).item()

    return prob


def yolo_detect_with_id(frame):
    boxes = []

    results = yolo.track(
        frame,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=IMG_SIZE,
        persist=True,
        tracker="inference/bytetrack.yaml",
        verbose=False
    )[0]

    if results.boxes:
        for box in results.boxes:
            # ✅ ONLY HUMAN CLASS
            if int(box.cls[0]) != 0 or box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])

            boxes.append((x1, y1, x2, y2, track_id))

    return boxes


def draw_boxes(frame, boxes):
    for x1, y1, x2, y2, tid in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{tid}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# =====================================================
# MAIN LOOP
# =====================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # STEP 1: CNN EVENT DECISION
    # -----------------------------
    prob = cnn_predict(frame)
    prob_history.append(prob)
    avg_prob = sum(prob_history) / len(prob_history)

    if avg_prob >= SNATCH_THRESHOLD:
        consecutive += 1
    else:
        consecutive = 0

    if consecutive >= CONSECUTIVE_FRAMES:
        snatch_detected = True

    # -----------------------------
    # STEP 2: YOLO DISPLAY WITH ID
    # -----------------------------
    boxes = yolo_detect_with_id(frame)
    draw_boxes(frame, boxes)

    # -----------------------------
    # STEP 3: SHOW EVENT RESULT
    # -----------------------------
    label = "CHAIN SNATCHING" if snatch_detected else "NOT-SNATCH"
    color = (0, 0, 255) if snatch_detected else (0, 255, 0)

    cv2.putText(
        frame,
        f"{label}  |  Prob: {avg_prob:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Chain Snatching Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =====================================================
# FINAL RESULT
# =====================================================
print("\n========== FINAL RESULT ==========")
if snatch_detected:
    print("🚨 VIDEO CLASSIFIED AS: CHAIN SNATCHING")
else:
    print("✅ VIDEO CLASSIFIED AS: NOT-SNATCH")
