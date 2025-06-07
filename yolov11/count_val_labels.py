import os
from pathlib import Path
from collections import defaultdict

# ✅ 기준 경로 (이 파일이 있는 위치 기준)
BASE_DIR = Path(__file__).resolve().parent
VAL_LABEL_DIR = BASE_DIR / "yolo_dataset" / "labels" / "val"

# 클래스별 카운터
class_counts = defaultdict(int)

# .txt 파일 순회
for file in os.listdir(VAL_LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    file_path = VAL_LABEL_DIR / file
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            class_id = line.strip().split()[0]
            class_counts[class_id] += 1

# 결과 출력
print("📊 검증 데이터셋 클래스별 개수 (labels/val 기준):")
total = 0
for class_id, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"클래스 {class_id}: {count}개")
    total += count

print(f"\n🔢 총 클래스 수: {len(class_counts)}개")
print(f"🧮 총 어노테이션 수: {total}개")