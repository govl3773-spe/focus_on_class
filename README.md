# focus_on_class develop

## 개요

`develop` 브랜치는 실험 브랜치에서 검증이 끝난 결과물만 모아 두는 통합 브랜치입니다.
현재는 데이터 수집, 데이터 분리, 모델 비교, 서비스 실행에 필요한 핵심 파일만 포함합니다.

## 브랜치 역할

- `1_capture`: 얼굴 crop 기반 데이터 수집과 품질 게이트 조정
- `2_minicnn/3_compare`: 여러 CNN 모델 비교와 최종 모델 선정 근거 정리
- `5_service`: FastAPI 기반 실시간 집중도 서비스 실행
- `develop`: 위 브랜치에서 검증된 산출물을 모아 최종 실행 흐름으로 정리

## 파일 역할

- `1-3_capture2.ipynb`: 얼굴 crop 수집과 품질 기준을 반영한 데이터 수집 노트북
- `dataset_123_split.ipynb`: 수집 데이터를 학습, 검증, 테스트 세트로 분리하는 노트북
- `1-7_compare.ipynb`: 여러 CNN 모델의 성능을 비교하는 주요 실험 노트북
- `1-8_all_model_compare.ipynb`: 비교 실험 결과를 확장해서 정리하는 노트북
- `2-1_three_model_compare.ipynb`: 최종 후보 3개 모델을 다시 비교하는 노트북
- `service/app.py`: 실시간 카메라 입력, 추론, 상태 판정을 제공하는 FastAPI 서비스
- `models/`: 최종 비교 후 남긴 서비스용 모델 가중치
- `mp_model/`: MediaPipe 얼굴 검출 및 랜드마크 추론에 필요한 모델 파일
- `pyproject.toml`, `uv.lock`: 실행 환경과 의존성 고정 파일

## docs 역할

`docs/` 폴더에는 `develop`에 포함된 흐름을 설명하는 문서만 우선 모아 둡니다.

- `docs/capture2_quality_collector_report.md`: `1_capture` 단계의 품질 수집 기준 설명
- `docs/report_compare1.md`: CNN 모델 비교 결과와 최종 추천 배경 정리
- `docs/model_selection_report.md`: 실시간 서비스 기준의 최종 모델 선정 근거
- `docs/GUIDE.md`: `service/app.py` 실행 방법과 서비스 동작 설명

## 현재 구조

```text
develop
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ mp_model/
├─ models/
├─ service/
│  └─ app.py
├─ docs/
│  ├─ capture2_quality_collector_report.md
│  ├─ report_compare1.md
│  ├─ model_selection_report.md
│  └─ GUIDE.md
├─ 1-3_capture2.ipynb
├─ dataset_123_split.ipynb
├─ 1-7_compare.ipynb
├─ 1-8_all_model_compare.ipynb
└─ 2-1_three_model_compare.ipynb
```

## 제외한 범위

- 초기 탐색 성격의 `0_prototype`
- threshold 조정과 OpenCV 실험 중심의 `4_threshold`
- 비교 과정에서만 쓰는 중간 산출물과 보조 실험 파일
