import cv2
import numpy as np
import os
import torch
from sam2.build_sam import build_sam2_video_predictor

# === 1. 프레임 로딩 ===
frames_dir = "videos/frames"
frame_files = sorted(
    [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
    key=lambda f: int(''.join(filter(str.isdigit, f)) or -1)
)
frames = [cv2.imread(os.path.join(frames_dir, f)) for f in frame_files]
if not frames:
    raise RuntimeError("No frames found in videos/frames")

frame_h, frame_w = frames[0].shape[:2]

# === 2. 모델 초기화 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = build_sam2_video_predictor(
    config_file="configs/sam2.1/sam2.1_hiera_t.yaml",
    ckpt_path="checkpoints/sam2.1_hiera_tiny.pt",
    device=device
)
inference_state = predictor.init_state(video_path=frames_dir)
predictor.reset_state(inference_state)

# === 3. GUI 상태 ===
window_name = "SAM2 Video Propagation"
cv2.namedWindow(window_name)
paused = True
current_idx = 0
click_pending = False
click_point = (0, 0)
obj_id = 1
video_segments = {}

# === 4. 마우스 콜백 ===
def on_mouse(event, x, y, flags, param):
    global click_pending, click_point
    if event == cv2.EVENT_LBUTTONDOWN and paused:
        click_point = (x, y)
        click_pending = True

cv2.setMouseCallback(window_name, on_mouse)

# === 5. 메인 루프 ===
while True:
    frame = frames[current_idx].copy()

    if click_pending:
        click_pending = False
        pt = np.array([[click_point]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=current_idx,
            obj_id=obj_id,
            points=pt,
            labels=lbl
        )
        print(f"[INFO] Propagating object {obj_id} from frame {current_idx}...")

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            for i, out_obj_id in enumerate(out_obj_ids):
                raw_mask = (out_mask_logits[i] > 0).cpu().numpy()
                if raw_mask.shape[0] == 1:
                    raw_mask = raw_mask[0]
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = {}
                video_segments[out_frame_idx][out_obj_id] = raw_mask

        print(f"[INFO] Done. Object {obj_id} now tracked in all frames.")
        obj_id += 1

    vis_frame = frame.copy()
    if current_idx in video_segments:
        for oid, mask in video_segments[current_idx].items():
            resized_mask = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            vis_frame[resized_mask] = vis_frame[resized_mask] * 0.4 + np.array([0, 255, 0]) * 0.6

    cv2.imshow(window_name, vis_frame)
    key = cv2.waitKey(100 if paused else 30) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):
        paused = not paused
    elif not paused:
        current_idx = (current_idx + 1) % len(frames)

cv2.destroyAllWindows()
