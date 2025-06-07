# YOLO + SAM2 통합 객체 추적 및 마스크 저장 버전

import cv2
import torch
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# === 설정 ===
sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_model = build_sam2(model_cfg, sam2_checkpoint).to(device)
predictor = SAM2ImagePredictor(sam_model, device=device)
yolo_model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

selected_ids = set()
thumbnail_height, thumbnail_width = 80, 80
next_object_id = 0
tracked_objects = []  # list of (id, box, class_id)
IOU_THRESHOLD = 0.7

SAVE_DIR = "saved_objects"
os.makedirs(SAVE_DIR, exist_ok=True)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (area1 + area2 - inter_area)


def apply_mask(frame, mask, color=(0, 255, 0), alpha=0.5):
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(bool)
    overlay = frame.copy()
    overlay[mask] = (np.array(color) * alpha + frame[mask] * (1 - alpha)).astype(np.uint8)
    return overlay


def save_selected_object(obj_id, frame, mask, class_name):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized * 255).astype(np.uint8)

    filename = f"{SAVE_DIR}/{now}_id{obj_id}_{class_name}"
    cv2.imwrite(filename + ".jpg", frame)
    cv2.imwrite(filename + "_mask.png", mask_resized)
    with open(filename + ".txt", "w") as f:
        f.write(f"object_id: {obj_id}\nclass: {class_name}\n")


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and x < thumbnail_width:
        idx = y // thumbnail_height
        if idx < len(display_thumbnails):
            obj_id = display_thumbnails[idx][0]
            if obj_id in selected_ids:
                selected_ids.remove(obj_id)
            elif len(selected_ids) < 2:
                selected_ids.add(obj_id)


cv2.namedWindow("YOLO + SAM2")
cv2.setMouseCallback("YOLO + SAM2", on_mouse)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    names = yolo_model.model.names

    current_tracked = []
    used_ids = set()
    for box, cls_id in zip(boxes, class_ids):
        matched_id = None
        for prev_id, prev_box, prev_cls in tracked_objects:
            if cls_id != prev_cls or prev_id in used_ids: continue
            if compute_iou(box, prev_box) > IOU_THRESHOLD:
                matched_id = prev_id
                break
        if matched_id is None:
            matched_id = next_object_id
            next_object_id += 1
        current_tracked.append((matched_id, box, cls_id))
        used_ids.add(matched_id)
    tracked_objects = current_tracked

    display_thumbnails = []
    for obj_id, box, cls_id in tracked_objects:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue
        thumb = cv2.resize(crop, (thumbnail_width, thumbnail_height))
        display_thumbnails.append((obj_id, thumb, (x1, y1, x2, y2), names[cls_id]))

    for obj_id, _, (x1, y1, x2, y2), cls_name in display_thumbnails:
        if obj_id not in selected_ids: continue
        predictor.set_image(frame_rgb)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
            multimask_output=False,
        )
        mask = masks[0][0]
        frame = apply_mask(frame, mask)
        save_selected_object(obj_id, frame, mask, cls_name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    panel_height = max(frame.shape[0], thumbnail_height * len(display_thumbnails))
    panel = np.zeros((panel_height, thumbnail_width, 3), dtype=np.uint8)
    for i, (obj_id, thumb, _, cls_name) in enumerate(display_thumbnails):
        y0 = i * thumbnail_height
        panel[y0:y0 + thumbnail_height] = thumb
        color = (0, 255, 0) if obj_id in selected_ids else (255, 255, 255)
        label = f"{i}: {cls_name}"
        cv2.putText(panel, label, (3, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    full_view = np.hstack([
        panel,
        np.pad(frame, ((0, panel_height - frame.shape[0]), (0, 0), (0, 0)), mode='constant')
    ])
    cv2.imshow("YOLO + SAM2", full_view)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
