
📦 YOLOv11 프로젝트 개요
이 프로젝트는 YOLOv11 기반 객체 탐지 모델 학습을 위한 전체 파이프라인을 제공합니다.
총 2개의 파이프라인이 존재하며, 각 파이프라인은 목적과 처리 흐름에 따라 다음과 같이 구분됩니다.

🚀 실행 방법 (How to Run)
🔑 실행 전, 실행 권한 부여 (최초 1회만 수행)
bash
복사
편집
chmod +x sml_eval_pipeline.sh
chmod +x augment_pipeline.sh
▶️ 파이프라인 실행
bash
복사
편집
./sml_eval_pipeline.sh      # s/m/l 모델 평가용
./augment_pipeline.sh       # 데이터 증강 기반 학습용
🧪 파이프라인 1: sml_eval_pipeline.sh
✅ 목적
YOLOv11의 사전 학습된 **세 가지 모델(s, m, l)**을 각각 학습 및 평가

원본 데이터만 사용하여 모델 성능 비교

📌 주요 작업 단계
모델 다운로드 (yolov11_model_download.py)

data.yaml 생성 (make_data_yaml.py)

YOLO 형식 변환 (convert_to_yolo.py)

검증 데이터 분리 (split_val.py)

모델 학습 (train_models.py)

성능 정리 (analyze_yolov11_results.py)

예측 결과 비교 (compare_wrong_predictions.py)

3개 모델의 앙상블 오답 분석 (ensemble_wrong_predictions.py)

🧠 특징
동일한 조건에서 모델 크기(s/m/l)별 비교 가능

예측 결과 분석 및 오류 클래스 비교 가능

🧪 파이프라인 2: augment_pipeline.sh
✅ 목적
클래스 불균형 해소 및 증강 중심의 학습 파이프라인

crop, collage, 색상 증강 포함하여 YOLO 학습 데이터를 준비

📌 주요 작업 단계
모델 다운로드 (yolov11_model_download.py)

data.yaml 생성 (make_data_yaml.py)

YOLO 형식 변환 (convert_to_yolo.py)

클래스 분포 분석 (analyze_class_imbalance.py)

클래스별 crop 증식 (crop_balancer.py)

콜라주 이미지 생성 (generate_collages.py)

YOLO 학습 포맷 변환 + 증강 (convert_with_aug.py)

모델 학습 (train_model.py)

성능 평가 (eval_model_aug.py)

🧠 특징
클래스 불균형 해소를 위한 crop 중심 증식

random 위치 + random 클래스 기반 collage 생성

Albumentations 라이브러리 기반 이미지 증강

YOLOv11-l 단일 모델 집중 학습

📁 프로젝트 구조 (요약)
bash
복사
편집
Project/
 └── yolov11/
     ├── scripts/             # 모든 파이썬 스크립트
     ├── model/               # yolov11s/m/l 모델 파일
     ├── yolo_dataset/        # YOLO 학습용 이미지/라벨
     ├── crops_data/          # 크롭된 이미지와 jsons
     ├── collage_images/      # 콜라주 이미지 결과
     ├── collage_json/        # 콜라주 json 어노테이션
     ├── requirements.txt     # 패키지 목록
     ├── sml_eval_pipeline.sh # s/m/l 평가용 파이프라인
     └── augment_pipeline.sh  # 증강 중심 파이프라인
