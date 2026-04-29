# 4_threshold

## 브랜치 목적

모델 결과를 안정화하기 위한 threshold 조정과 OpenCV 실시간 테스트를 분리해서 관리하는 브랜치입니다.

## `3-1_openCV_test.ipynb` 생성 이유

`3_compare` 브랜치의 `1-7_compare.ipynb`에서 OpenCV 실시간 실행과 화면 표시 성격의 코드를 별도 테스트 노트북으로 분리하기 위해 만들었습니다.

## OpenCV 테스트 분리 이유

- 모델 비교 노트북과 실시간 카메라 테스트 노트북의 목적이 다릅니다.
- 비교 실험 로직과 OpenCV 디버깅 로직을 분리해야 수정 범위를 줄일 수 있습니다.
- threshold 조정 시 실시간 반응만 빠르게 확인할 수 있어 반복 실험이 쉬워집니다.

## threshold 조정 작업 방향

- `3-1_openCV_test.ipynb`에서 실시간 예측 안정성만 집중 점검합니다.
- `confidence_threshold`, `state_ratio_threshold` 같은 안정화 파라미터를 이 브랜치에서만 조정합니다.
- 모델 비교용 원본 노트북은 유지하고, threshold 관련 반복 실험은 이 브랜치에서만 진행합니다.
