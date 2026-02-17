import cv2
import os

def convert(video_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    for video in os.listdir(video_folder):
        cap = cv2.VideoCapture(os.path.join(video_folder, video))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % 5 == 0:  
                cv2.imwrite(f"{save_folder}/{video}_{count}.jpg", frame)
            count += 1
        cap.release()

convert("videos/snatch", "dataset/train/snatch")
convert("videos/not_snatch", "dataset/train/not_snatch")
