# 4_threshold

## 브랜치 목적

모델 결과를 안정화하기 위한 threshold 조정과 OpenCV 실시간 테스트를 분리해서 관리하는 브랜치입니다.

## 포함 파일

- `3-1_openCV_test.ipynb`
- `models/resnet18_best.pt`
- `pyproject.toml`
- `uv.lock`
- `README.md`

## `3-1_openCV_test.ipynb` 생성 이유

`3_compare` 브랜치의 비교 노트북에서 OpenCV 실시간 테스트 부분만 분리해 threshold 조정 전용으로 관리하기 위해 만들었습니다.

## 작업 방향

- 실시간 예측 안정성 확인
- `confidence_threshold`, `state_ratio_threshold` 조정
- OpenCV 테스트와 비교 실험의 책임 분리
