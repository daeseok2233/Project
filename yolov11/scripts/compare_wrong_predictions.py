import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from ultralytics import YOLO
import random

# ✅ 현재 스크립트 기준 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PILL_LIST_PATH = os.path.join(BASE_DIR, "pill_list2.txt")
IMAGE_DIR = os.path.join(BASE_DIR, "yolo_dataset", "images", "train")
LABEL_DIR = os.path.join(BASE_DIR, "yolo_dataset", "labels", "train")
MODEL_PATHS = {
    "YOLOv11-s": os.path.join(BASE_DIR, "runs", "yolov11s", "exp", "weights", "best.pt"),
    "YOLOv11-m": os.path.join(BASE_DIR, "runs", "yolov11m", "exp", "weights", "best.pt"),
    "YOLOv11-l": os.path.join(BASE_DIR, "runs", "yolov11l", "exp", "weights", "best.pt"),
}

# ✅ 한글 폰트 설정 (리눅스 기준, 윈도우면 수정 필요)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_prop = fm.FontProperties(fname=font_path) if os.path.exists(font_path) else None
mpl.rcParams['axes.unicode_minus'] = False

# ✅ 클래스 이름 로드
with open(PILL_LIST_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# ✅ 랜덤 색상 생성
random.seed(42)
colors = [(random.random(), random.random(), random.random()) for _ in class_names]

# ✅ 모델별 예측
for model_name, model_path in MODEL_PATHS.items():
    if not os.path.exists(model_path):
        print(f"❌ {model_name} 모델 파일 없음: {model_path}")
        continue

    print(f"\n🔍 {model_name} 예측 시작\n" + "-" * 50)
    model = YOLO(model_path)

    results = model.predict(source=IMAGE_DIR, conf=0.5, iou=0.5, agnostic_nms=True, save=False, verbose=False)
    total_wrong = 0

    for result in results:
        img_name = os.path.basename(result.path)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

        pred_classes = [int(c) for c in result.boxes.cls.cpu().numpy()]
        pred_labels = [class_names[i] for i in pred_classes]

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                gt_classes = [int(line.strip().split()[0]) for line in f.readlines()]
            gt_labels = [class_names[i] for i in sorted(gt_classes)]
        else:
            gt_labels = []

        if sorted(pred_labels) != gt_labels:
            total_wrong += 1
            print(f"\n🖼️ 이미지: {img_name}")
            print(f"📌 예측 클래스: {sorted(pred_labels)}")
            print(f"✅ 정답 클래스: {gt_labels}")

            img_path = os.path.join(IMAGE_DIR, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            ax = plt.gca()

            for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                cls = pred_classes[i]
                label = class_names[cls]
                conf = float(result.boxes.conf[i].cpu().numpy()) * 100
                label_text = f"{label} {conf:.1f}%"
                color = colors[cls]
                x1, y1, x2, y2 = box

                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, facecolor='none', linewidth=2)
                ax.add_patch(rect)

                ax.text(x1, y1 - 5, label_text, color=color, fontsize=10,
                        fontproperties=font_prop,
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

            plt.axis('off')
            plt.title(f"{model_name} 틀린 예측", fontproperties=font_prop)
            plt.show()

    print(f"\n❌ {model_name} - 틀린 이미지 개수: {total_wrong}장")