# 1_capture

## 브랜치 목적

MediaPipe 기반 데이터 수집과 얼굴 crop 생성, 후속 모델 학습용 입력 데이터를 준비하는 브랜치입니다.

## 주요 구성 요소

- `mp_model/`: MediaPipe 관련 모델 및 수집 파이프라인에서 사용하는 자산
- `1-3_capture2.ipynb`: 얼굴 crop 데이터 수집과 품질 보정 로직을 포함한 핵심 캡처 노트북

## develop에 포함되는 파일

- `mp_model/`
- `1-3_capture2.ipynb`

## develop 반영 원칙

`develop`에는 최종 테스트에 필요한 캡처 자산만 반영하고, 나머지 실험 노트북, 임시 데이터, 캐시 파일은 포함하지 않습니다.
