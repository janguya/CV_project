import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SAM2 Video Predictor
video_predictor = build_sam2_video_predictor().to(device)

# Video capture
cap = cv2.VideoCapture('videos/bedroom.mp4')
paused = False
clicked_points = []

# Mouse callback function
def click_event(event, x, y, flags, params):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and paused:
        clicked_points.append([x, y])

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_event)

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display_frame = frame.copy()
        for point in clicked_points:
            masks, _, _ = video_predictor.predict(frame, np.array([point]), np.array([1]))
            mask = masks[0].astype(np.uint8) * 255
            color_mask = np.zeros_like(frame)
            color_mask[mask == 255] = [0, 255, 0]  # Green mask
            display_frame = cv2.addWeighted(display_frame, 1, color_mask, 0.5, 0)

        cv2.imshow('Frame', display_frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord(' '):
        paused = not paused
    elif key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
