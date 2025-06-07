import cv2
import torch
import numpy as np
import torchvision
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time

prev_time = time.time()
last_print_time = prev_time

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

# === SETUP ===
sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

sam_model = build_sam2(model_cfg, sam2_checkpoint).to(device)
predictor = SAM2ImagePredictor(sam_model, device=device)

yolo_model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
obj_id_counter = 1

def apply_mask(frame, mask, color=(0, 255, 0), alpha=0.5):
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(bool)

    overlay = frame.copy()
    overlay[mask] = (np.array(color) * alpha + frame[mask] * (1 - alpha)).astype(np.uint8)
    return overlay


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # --- YOLO 결과 콘솔 출력 (1초마다) ---
    now = time.time()
    if now - last_print_time >= 1.0:
        print("[Detected Objects]")
        for box, cls_id in zip(results[0].boxes.xyxy.cpu().numpy(),
                               results[0].boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            class_name = yolo_model.model.names[int(cls_id)]
            print(f"- Class: {class_name}, Box: ({x1}, {y1}, {x2}, {y2})")
        print("===")
        last_print_time = now

    for i, box in enumerate(boxes[:2]):
        x1, y1, x2, y2 = map(int, box)
        input_image = frame_rgb.copy()
        box_tensor = torch.tensor([[x1, y1, x2, y2]], device=device)

        predictor.set_image(input_image)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_tensor.cpu().numpy(),
            multimask_output=False,
        )

        mask = masks[0][0]
        frame = apply_mask(frame, mask)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- FPS 계산 및 표시 ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO + SAM2 Segmentation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
