#!/bin/bash

cd /workspace/Project/yolov11

echo "🔹 [0] 패키지 설치 시작..."
pip install -r requirements.txt

echo "🔹 [1] 모델 다운로드"
python yolov11_model_download.py

echo "🔹 [2] data.yaml 생성"
python make_data_yaml.py

echo "🔹 [3] YOLO 포맷 변환"
python convert_to_yolo.py

echo "🔹[4] 라벨 수 확인"
python count_val_labels.py

echo "🔹[5] 모델 학습 시작"
python train_models.py

echo "🔹[6] 결과 분석"
python analyze_yolov11_results.py

echo "🔹[7] 잘못된 예측 비교 (선택)"
python compare_wrong_predictions.py

echo "🔹[8] 앙상블 평가 (선택)"
python ensemble_wrong_predictions.py
