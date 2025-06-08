import os
import cv2
import torch
import numpy as np
import threading
from sam2.build_sam import build_sam2_video_predictor

# === 설정 ===

frames_dir = "videos/frames"
sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 프레임 불러오기 ===
frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                     key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))
frames = [cv2.imread(os.path.join(frames_dir, f)) for f in frame_files]
total_frames = len(frames)
assert total_frames > 0, f"No frames found in {frames_dir}"

# === SAM2 초기화 ===
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
inference_state = predictor.init_state(video_path=frames_dir)
predictor.reset_state(inference_state)

# === 변수 초기화 ===
current_idx = 0
paused = False
selected_points = {}
selected_labels = {}
next_obj_id = 1
colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]
video_segments = {}
is_propagating = False

# === 마우스 이벤트 ===
def on_mouse(event, x, y, flags, param):
    global next_obj_id, selected_points, selected_labels
    if event == cv2.EVENT_LBUTTONDOWN and paused and len(selected_points) < 3:
        if is_propagating:
            print("Propagation in progress, please wait.")
            return
        obj_id = next_obj_id
        next_obj_id += 1
        selected_points[obj_id] = [(x, y)]
        selected_labels[obj_id] = [1]
        threading.Thread(target=propagate_async, args=(obj_id, [(x, y)], [1])).start()

cv2.namedWindow("VideoSegmentation")
cv2.setMouseCallback("VideoSegmentation", on_mouse)

# === 마스크 합성 ===
def overlay_mask(frame, mask, color, alpha=0.5):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=0)
    if mask.shape != frame.shape[:2]:
        try:
            mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        except:
            return frame
    mask_bool = mask.astype(bool)
    overlay = frame.copy().astype(np.float32)
    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
    return overlay.astype(np.uint8)

# === propagate 비동기 처리 ===
def propagate_async(obj_id, points, labels):
    global is_propagating
    is_propagating = True
    try:
        predictor.reset_state(inference_state)
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=current_idx,
            obj_id=obj_id,
            points=points_np,
            labels=labels_np,
        )
        for f_idx, out_obj_ids, out_masks in predictor.propagate_in_video(inference_state=inference_state):
            for i, oid in enumerate(out_obj_ids):
                if oid == obj_id:
                    video_segments.setdefault(oid, {})[f_idx] = (out_masks[i] > 0.0).cpu().numpy().squeeze()
    finally:
        is_propagating = False

# === 메인 루프 ===
while True:
    frame = frames[current_idx].copy()
    for obj_id, masks in video_segments.items():
        if current_idx in masks:
            frame = overlay_mask(frame, masks[current_idx], color=colors[(obj_id - 1) % len(colors)])

    cv2.imshow("VideoSegmentation", frame)
    key = cv2.waitKey(30 if not paused else 100) & 0xFF
    if key == 27:
        break
    elif key == ord(' '):
        paused = not paused

    if not paused :
        current_idx = (current_idx + 1) % total_frames

cv2.destroyAllWindows()
