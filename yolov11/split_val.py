import os
import shutil
import random
from collections import defaultdict
from pathlib import Path

# ✅ 기준 경로 (이 파일이 있는 위치 기준)
BASE_DIR = Path(__file__).resolve().parent
YOLO_DIR = BASE_DIR / "yolo_dataset"

label_dir = YOLO_DIR / "labels" / "train"
image_dir = YOLO_DIR / "images" / "train"
label_val_dir = YOLO_DIR / "labels" / "val"
image_val_dir = YOLO_DIR / "images" / "val"

os.makedirs(label_val_dir, exist_ok=True)
os.makedirs(image_val_dir, exist_ok=True)

# 이미지 경로 찾기 함수 (확장자 자동 탐색)
def find_image_path(base_name, image_dir):
    for ext in ['.png', '.jpg', '.jpeg']:
        path = image_dir / (base_name + ext)
        if path.exists():
            return path
    return None

# 클래스별 파일 목록 수집
class_to_files = defaultdict(list)

for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue

    file_path = label_dir / file
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            class_id = line.strip().split()[0]
            class_to_files[class_id].append(file)
            break  # 하나의 클래스 기준만 등록

# 클래스별로 일정 비율을 val로 분할
selected_for_val = set()
for class_id, files in class_to_files.items():
    files = list(set(files))  # 중복 제거
    random.shuffle(files)
    val_count = max(1, int(len(files) * 0.2))
    selected_for_val.update(files[:val_count])

# 이미지 + 라벨 이동
moved_count = 0
for file in selected_for_val:
    base = os.path.splitext(file)[0]
    label_src = label_dir / file
    label_dst = label_val_dir / file

    image_src = find_image_path(base, image_dir)
    if image_src is None or not label_src.exists():
        print(f"❌ 이동 실패 (파일 없음): {base}")
        continue

    image_dst = image_val_dir / image_src.name

    shutil.move(str(label_src), str(label_dst))
    shutil.move(str(image_src), str(image_dst))
    moved_count += 1

print(f"✅ val 데이터셋으로 이동된 샘플 수: {moved_count}개")