📦 YOLOv11 프로젝트 개요

이 프로젝트는 YOLOv11 기반의 객체 탐지 모델 학습을 위한 전체 파이프라인을 제공합니다. 두 가지 주요 파이프라인이 존재하며, 각각의 목적과 처리 흐름에 따라 나뉩니다.


## 🚀 실행 방법 (How to Run)

본 프로젝트는 아래 명령어 한 줄로 전체 파이프라인을 자동 실행할 수 있습니다:

./sml_eval_pipeline.sh    ./augment_pipeline.sh       

📌 실행 전 한 번만 아래 명령어로 실행 권한을 부여해주세요:

chmod +x sml_eval_pipeline.sh   chmod +x augment_pipeline.sh


🧪 파이프라인 1: sml_eval_pipeline.sh

✅ 목적

세 가지 YOLOv11 사전학습 모델(s, m, l)을 각각 학습 및 비교 평가

증강 없이 원본 데이터를 기반으로 성능 비교

🔧 주요 작업 단계

YOLOv11 모델 다운로드 (yolov11_model_download.py)

data.yaml 생성 (make_data_yaml.py)

초기 YOLO 형식 변환 (convert_to_yolo.py)

검증 데이터 분리 (split_val.py)

모델별 학습 (train_models.py)

모델별 평가 결과 정리 (analyze_yolov11_results.py)

모델별 예측 오류 비교 (compare_wrong_predictions.py)

3개 모델 앙상블 오류 분석 (ensemble_wrong_predictions.py)

🧪 특징

동일한 데이터셋으로 3개의 YOLOv11 파라미터 비교 (s, m, l)

모델 간 성능 편차 확인 및 예측 오류 비교 가능

앙상블 오류 분석으로 성능 보완 가능성 탐색

🚀 파이프라인 2: augment_pipeline.sh

✅ 목적

클래스 불균형 해소를 위한 증강 중심의 데이터 파이프라인

크롭, 콜라주, 색상 증강을 포함한 YOLO 학습 데이터 준비 및 학습 수행

🔧 주요 작업 단계

YOLOv11 모델 다운로드

data.yaml 생성

초기 YOLO 형식 변환 (convert_to_yolo.py)

클래스별 등장 횟수 분석 (analyze_class_imbalance.py)

클래스 불균형에 따라 crop_balancer.py 실행

크롭된 이미지로 콜라주 이미지 생성 (generate_collages.py)

콜라주 이미지 증강 및 YOLO 포맷 변환 (convert_with_aug.py)

YOLO 모델 학습 (train_model.py)

모델 검증 및 평가 (eval_model_aug.py)

🧪 특징

클래스 등장 횟수 균등화를 위한 crop 증식

기존 학습 이미지 Crop + 랜덤한 클래스 / 랜덤한 위치 : 콜라주 생성

Albumentations 기반 색상 증강 적용

모델 학습은 하나의 YOLOv11 모델(yolo11l.pt)에 집중

![image](https://github.com/user-attachments/assets/eb52c6dc-5a79-4b02-abcb-66cea48f40b5)
