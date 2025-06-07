import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# === 설정 ===
sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 모델 초기화 ===
sam_model = build_sam2(model_cfg, sam2_checkpoint).to(device)
predictor = SAM2ImagePredictor(sam_model, device=device)
yolo_model = YOLO("yolov8n.pt")

# === 비디오 스트림 시작 ===
cap = cv2.VideoCapture(0)

# === 썸네일 선택 상태 ===
selected_indices = set()
thumbnail_height, thumbnail_width = 80, 80
thumbnail_boxes = []

def apply_mask(frame, mask, color=(0, 255, 0), alpha=0.5):
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(bool)
    overlay = frame.copy()
    overlay[mask] = (np.array(color) * alpha + frame[mask] * (1 - alpha)).astype(np.uint8)
    return overlay

# === 마우스 콜백 ===
def on_mouse(event, x, y, flags, param):
    global selected_indices
    if event == cv2.EVENT_LBUTTONDOWN and x < thumbnail_width:
        idx = y // thumbnail_height
        if idx < len(thumbnail_boxes):
            if idx in selected_indices:
                selected_indices.remove(idx)
            elif len(selected_indices) < 2:
                selected_indices.add(idx)

cv2.namedWindow("YOLO + SAM2")
cv2.setMouseCallback("YOLO + SAM2", on_mouse)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = yolo_model.model.names

    # === 썸네일 리스트 생성 ===
    thumbnails = []
    thumbnail_boxes = []
    for i, (box, cls_id) in enumerate(zip(boxes, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        thumb = cv2.resize(crop, (thumbnail_width, thumbnail_height))
        thumbnails.append(thumb)
        thumbnail_boxes.append((x1, y1, x2, y2, class_names[cls_id]))

    # === SAM2로 선택된 객체만 마스크 추출 ===
    for idx in selected_indices:
        if idx >= len(thumbnail_boxes): continue
        x1, y1, x2, y2, _ = thumbnail_boxes[idx]
        box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        predictor.set_image(frame_rgb)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np,
            multimask_output=False,
        )
        mask = masks[0][0]
        frame = apply_mask(frame, mask)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # === 썸네일 패널 만들기 ===
    panel_h = max(frame.shape[0], thumbnail_height * len(thumbnails))
    panel = np.zeros((panel_h, thumbnail_width, 3), dtype=np.uint8)
    for i, (thumb, (_, _, _, _, cls_name)) in enumerate(zip(thumbnails, thumbnail_boxes)):
        y0 = i * thumbnail_height
        panel[y0:y0+thumbnail_height, :thumbnail_width] = thumb
        label = f"{i}: {cls_name}"
        color = (0,255,0) if i in selected_indices else (255,255,255)
        cv2.putText(panel, label, (2, y0+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # === 화면 병합 및 출력 ===
    pad_h = max(panel.shape[0], frame.shape[0])
    frame_padded = np.zeros((pad_h, frame.shape[1], 3), dtype=np.uint8)
    frame_padded[:frame.shape[0], :frame.shape[1]] = frame
    full_view = np.hstack([panel, frame_padded])
    cv2.imshow("YOLO + SAM2", full_view)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
