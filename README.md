# focus_on_class develop

## 1. 프로젝트 개요

이 브랜치는 최종 테스트를 위한 통합 브랜치입니다. 데이터 수집 단계에서 필요한 MediaPipe 자산과 모델 비교 단계에서 필요한 노트북/모델 파일만 선별해서 포함합니다.

## 2. 전체 폴더 구조

```text
develop
├─ pyproject.toml
├─ uv.lock
├─ README.md
├─ mp_model/
├─ models/
├─ 1-3_capture2.ipynb
├─ dataset_123_split.ipynb
├─ 1-7_compare.ipynb
├─ 1-8_all_model_compare.ipynb
└─ 2-1_three_model_compare.ipynb
```

## 3. 데이터 수집 흐름

1. `mp_model/`의 MediaPipe task 파일을 사용해 얼굴/포즈 관련 입력을 준비합니다.
2. `1-3_capture2.ipynb`에서 얼굴 crop 기반 수집 및 품질 확인 로직을 실행합니다.
3. 수집 결과는 후속 학습 및 비교 실험의 입력 데이터로 사용합니다.

## 4. 모델 비교 흐름

1. `dataset_123_split.ipynb`로 비교용 데이터를 `train/test/val` 구조로 분리합니다.
2. `1-8_all_model_compare.ipynb`에서 여러 CNN 계열 모델을 공통 파이프라인으로 학습/비교합니다.
3. `1-7_compare.ipynb`에서 저장된 모델을 같은 테스트셋 기준으로 평가합니다.
4. `2-1_three_model_compare.ipynb`에서 최종 후보 3개 모델을 조건을 맞춰 다시 비교합니다.

## 5. 실행 순서

1. `1-3_capture2.ipynb`
2. `dataset_123_split.ipynb`
3. `1-8_all_model_compare.ipynb`
4. `1-7_compare.ipynb`
5. `2-1_three_model_compare.ipynb`

## 6. 각 노트북 역할

- `1-3_capture2.ipynb`: 얼굴 crop 데이터 수집 및 품질 보정
- `dataset_123_split.ipynb`: 비교용 데이터셋 분리
- `1-8_all_model_compare.ipynb`: 다중 모델 학습/비교
- `1-7_compare.ipynb`: 저장 모델 공통 평가
- `2-1_three_model_compare.ipynb`: 최종 3개 모델 비교

## 7. 현재 develop 브랜치에 포함된 파일 목록

- `pyproject.toml`
- `uv.lock`
- `mp_model/`
- `1-3_capture2.ipynb`
- `models/`
- `dataset_123_split.ipynb`
- `1-7_compare.ipynb`
- `1-8_all_model_compare.ipynb`
- `2-1_three_model_compare.ipynb`

## 8. 제외된 브랜치와 제외 이유

- `0_prototype`: 초기 프로토타입 실험용이므로 `develop`에 포함하지 않음
- `2_minicnn`: 개별 모델 테스트용이므로 `develop`에 포함하지 않음
- `4_threshold`: threshold 조정 및 OpenCV 테스트 분리 전용 브랜치로 별도 worktree에서 진행
