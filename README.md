# 3_compare

## 브랜치 목적

여러 분류 모델의 학습 결과를 동일한 기준으로 비교하고, 데이터 분리 및 성능 비교 실험을 정리하는 브랜치입니다.

## models 폴더 역할

`models/` 폴더는 비교 대상 모델의 체크포인트와 학습 이력 CSV를 보관하는 위치입니다. `develop`에는 최종 비교/테스트에 필요한 모델 파일만 포함합니다.

## 각 노트북 역할

- `dataset_123_split.ipynb`: 원본 `dataset_123` 데이터를 `train/test/val` 구조로 복사 분리하는 노트북
- `1-7_compare.ipynb`: 저장된 `*_best.pt` 모델들을 같은 테스트셋 기준으로 평가하는 비교 노트북
- `1-8_all_model_compare.ipynb`: 공통 학습 파이프라인으로 여러 CNN 계열 모델을 반복 학습/비교하는 노트북
- `2-1_three_model_compare.ipynb`: MobileNet, EfficientNet, ResNet18 세 모델만 선별해 가중치 조건을 맞춰 비교하는 노트북

## develop에 포함되는 파일

- `models/`
- `dataset_123_split.ipynb`
- `1-7_compare.ipynb`
- `1-8_all_model_compare.ipynb`
- `2-1_three_model_compare.ipynb`

## develop 반영 원칙

`develop`에는 최종 테스트에 필요한 비교 자산만 반영하고, OpenCV 분리 테스트 노트북과 기타 실험/임시 산출물은 포함하지 않습니다.
