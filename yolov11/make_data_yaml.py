from pathlib import Path

# ✅ 기준 경로
BASE_DIR = Path(__file__).resolve().parent
PILL_LIST_PATH = BASE_DIR / "pill_list2.txt"
YAML_OUTPUT_PATH = BASE_DIR / "yolo_dataset" / "data.yaml"

# 클래스 이름 추출
class_names = []
with open(PILL_LIST_PATH, encoding="utf-8") as f:
    for line in f:
        name = line.strip()
        if not name:
            continue
        if "-" in name:
            name = name.split("-")[0].strip()  # "이름 -숫자" → 이름만 추출
        class_names.append(name)

# ✅ 절대 경로로 수정
absolute_path = YAML_OUTPUT_PATH.parent.resolve()

# YAML 파일 쓰기
with open(YAML_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"path: {absolute_path}\n")  # ✅ 절대 경로로 작성
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write("names:\n")
    for name in class_names:
        f.write(f"  - {name}\n")

print(f"✅ data.yaml 생성 완료 → {YAML_OUTPUT_PATH}")