from pathlib import Path
from ultralytics import YOLO

# ✅ 저장할 위치: yolov11/model/
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ✅ 다운로드할 모델 리스트 (s, m, l만)
model_names = ["yolo11s.pt", "yolo11m.pt", "yolo11l.pt"]

available_models = []

for name in model_names:
    try:
        print(f"🔍 다운로드 시도: {name}")
        model = YOLO(name)  # 캐시에 다운로드

        # ✅ 실제 저장된 캐시 경로에서 model/ 폴더로 복사
        src_path = model.ckpt_path if hasattr(model, "ckpt_path") else None
        dst_path = MODEL_DIR / name

        if src_path and Path(src_path).exists():
            dst_path.write_bytes(Path(src_path).read_bytes())
            print(f"✅ 저장 완료: {dst_path.name}")
            available_models.append(name)
        else:
            print(f"⚠️ 원본 파일 경로 없음: {src_path}")

    except Exception as e:
        print(f"❌ 다운로드 실패: {name} → {e}")

# ✅ 결과 출력
print("\n✅ 최종 저장된 YOLOv11 모델 목록:")
for m in available_models:
    print(" -", m)