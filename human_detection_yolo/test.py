import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from collections import deque

# -------------------------
# 1. DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 2. MODEL (SAME AS TRAINING)
# -------------------------
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
        x = self.features(x)
        return self.classifier(x)

model = CNN().to(device)

# -------------------------
# 3. LOAD TRAINED MODEL
# -------------------------
model_path = "models/chain_snatch/epoch_5.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Loaded model:", model_path)

# -------------------------
# 4. IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -------------------------
# 5. VIDEO PATH
# -------------------------
video_path = "videos_test/chain.webm"   # change if needed

if not os.path.exists(video_path):
    raise FileNotFoundError("❌ Video not found")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video file")

# -------------------------
# 6. PARAMETERS
# -------------------------
SNATCH_THRESHOLD = 0.6        # probability threshold
CONSECUTIVE_FRAMES = 5        # frames to confirm event
SMOOTHING_WINDOW = 5          # probability smoothing

prob_history = deque(maxlen=SMOOTHING_WINDOW)

frame_count = 0
snatch_frames = 0
consecutive_snatch = 0
snatch_event_detected = False
snatch_start_frame = None
snatch_probs = []             # for confidence calculation

# -------------------------
# 7. FRAME BY FRAME ANALYSIS
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert OpenCV -> PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    # --- SMOOTHING ---
    prob_history.append(prob)
    avg_prob = sum(prob_history) / len(prob_history)

    # --- CLASSIFICATION ---
    if avg_prob >= SNATCH_THRESHOLD:
        label = "SNATCH"
        color = (0, 0, 255)
        snatch_frames += 1
        consecutive_snatch += 1
        snatch_probs.append(avg_prob)

        if consecutive_snatch == CONSECUTIVE_FRAMES:
            snatch_event_detected = True
            snatch_start_frame = frame_count - CONSECUTIVE_FRAMES + 1
    else:
        label = "NOT-SNATCH"
        color = (0, 255, 0)
        consecutive_snatch = 0

    print(f"Frame {frame_count} → {label} ({avg_prob:.3f})")

    # Display
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Chain Snatching Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------
# 8. FINAL RESULT
# -------------------------
print("\n========== FINAL RESULT ==========")
print("Total frames      :", frame_count)
print("Snatch frames     :", snatch_frames)

if snatch_event_detected:
    snatch_confidence = (sum(snatch_probs) / len(snatch_probs)) * 100
    print("🚨 VIDEO CLASSIFIED AS: CHAIN SNATCHING")
    print("⏱ Snatching started near frame:", snatch_start_frame)
    print(f"📊 Snatch Confidence : {snatch_confidence:.2f}%")
else:
    print("✅ VIDEO CLASSIFIED AS: NOT-SNATCH")
