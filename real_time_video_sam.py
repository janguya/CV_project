import cv2
import torch
import numpy as np
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_model = build_sam2(model_cfg, sam2_checkpoint).to(device)
predictor = SAM2ImagePredictor(sam_model, device=device)

cap = cv2.VideoCapture(0)

selected_points = []
selected_ids = []
next_obj_id = 1
filter_modes = {}  # filter (0: none, 1: white fill, 2: alpha mask, 3: edge)

WINDOW_NAME = "SAM2 Live Select"
cv2.namedWindow(WINDOW_NAME)

last_seg_time = 0
seg_interval = 0.5
latest_masks = []

THUMB_SIZE = 64
THUMB_MARGIN = 10

def draw_delete_icon(frame, x, y):
    center = (x + 8, y + 8)
    cv2.circle(frame, center, 9, (0, 0, 255), -1)
    cv2.line(frame, (x + 5, y + 5), (x + 11, y + 11), (255, 255, 255), 2)
    cv2.line(frame, (x + 11, y + 5), (x + 5, y + 11), (255, 255, 255), 2)

def draw_filter_button(frame, x, y, mode):
    color = [(50, 50, 50), (255, 255, 255), (200, 200, 255), (0, 255, 255)][mode % 4]
    cv2.rectangle(frame, (x, y), (x + THUMB_SIZE, y + THUMB_SIZE), color, 2)
    cv2.putText(frame, f"F{mode}", (x + 5, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def on_mouse(event, x, y, flags, param):
    global selected_points, selected_ids, next_obj_id
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, (_, _, obj_id, _) in enumerate(latest_masks):
            ty = THUMB_MARGIN + idx * (THUMB_SIZE + THUMB_MARGIN)
            del_x1, del_y1 = THUMB_MARGIN + THUMB_SIZE - 18, ty + 2
            del_x2, del_y2 = del_x1 + 16, del_y1 + 16
            if del_x1 <= x < del_x2 and del_y1 <= y < del_y2:
                selected_points = [pt for pt, oid in zip(selected_points, selected_ids) if oid != obj_id]
                selected_ids = [oid for oid in selected_ids if oid != obj_id]
                filter_modes.pop(obj_id, None)
                return
            thumb_x1, thumb_y1 = THUMB_MARGIN, ty
            thumb_x2, thumb_y2 = thumb_x1 + THUMB_SIZE, thumb_y1 + THUMB_SIZE
            if thumb_x1 <= x < thumb_x2 and thumb_y1 <= y < thumb_y2:
                filter_modes[obj_id] = (filter_modes.get(obj_id, 0) + 1) % 4
                return
        if len(set(selected_ids)) >= 2:
            removed_id = selected_ids.pop(0)
            selected_points.pop(0)
            filter_modes.pop(removed_id, None)
        selected_points.append((x, y))
        selected_ids.append(next_obj_id)
        next_obj_id += 1


cv2.setMouseCallback(WINDOW_NAME, on_mouse)


def apply_mask(frame, mask, mode):
    mask = mask.astype(np.uint8)
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask.astype(bool)
    if mode == 1:
        frame[mask_bool] = [255, 255, 255]
    elif mode == 2:
        alpha_overlay = frame.copy()
        alpha_overlay[mask_bool] = alpha_overlay[mask_bool] * 0.4 + np.array([255, 255, 255]) * 0.6
        frame[mask_bool] = alpha_overlay[mask_bool]
    elif mode == 3:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)
    return frame


print("[INFO] 시작: 좌클릭으로 객체 추가, 썸네일 클릭 시 필터 순환, '-' 클릭 시 제거")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = time.time()

    if now - last_seg_time >= seg_interval:
        predictor.set_image(frame_rgb)
        latest_masks = []
        for (point, obj_id) in zip(selected_points, selected_ids):
            coords = np.array([point], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            try:
                masks, _, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=False)
                mask = masks[0]
                region = frame.copy()
                mask_resized = mask.astype(np.uint8)
                masked_region = cv2.bitwise_and(region, region, mask=mask_resized)
                x_indices = np.any(mask_resized, axis=0)
                y_indices = np.any(mask_resized, axis=1)
                if np.any(x_indices) and np.any(y_indices):
                    x1, x2 = np.where(x_indices)[0][[0, -1]]
                    y1, y2 = np.where(y_indices)[0][[0, -1]]
                    cropped = masked_region[y1:y2, x1:x2]
                    thumbnail = cv2.resize(cropped, (THUMB_SIZE, THUMB_SIZE)) if cropped.size > 0 else np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
                else:
                    thumbnail = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
                latest_masks.append((mask, point, obj_id, thumbnail))
            except Exception as e:
                print(f"[ERROR] SAM2 failed on point {point}: {e}")
        last_seg_time = now

    vis_frame = frame.copy()
    for idx, (mask, point, obj_id, thumb) in enumerate(latest_masks):
        vis_frame = apply_mask(vis_frame, mask, mode=filter_modes.get(obj_id, 0))
        cv2.circle(vis_frame, point, 5, (0, 255, 0), -1)
        ty = THUMB_MARGIN + idx * (THUMB_SIZE + THUMB_MARGIN)
        vis_frame[ty : ty + THUMB_SIZE, THUMB_MARGIN : THUMB_MARGIN + THUMB_SIZE] = thumb
        draw_filter_button(vis_frame, THUMB_MARGIN, ty, filter_modes.get(obj_id, 0))
        draw_delete_icon(vis_frame, THUMB_MARGIN + THUMB_SIZE - 18, ty + 2)

    cv2.imshow(WINDOW_NAME, vis_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
