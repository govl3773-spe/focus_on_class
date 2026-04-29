# 2_minicnn

## 브랜치 목적

Mini CNN, MobileNet, ResNet 계열의 개별 모델 학습과 전이학습 설정을 테스트하기 위한 브랜치입니다.

## 테스트한 모델 및 실험 내용 요약

- `mobilenet/mobilenet.ipynb`: MobileNet 기반 학습 및 추론 실험
- `resnet/resnet_test.ipynb`: ResNet 전이학습 및 데이터 분리 실험
- `mobilenet/checkpoints/`, `resnet/*.pt`: 모델별 체크포인트 실험 결과
- `mobilenet_dataset/`, `resnet_dataset/`, `dataset/`: 모델 테스트용 데이터셋 구성
- `docs/transfer-learning-opencv-workflow.md`: 전이학습 및 OpenCV 테스트 흐름 정리

## develop에 포함하지 않는 이유

이 브랜치는 개별 모델 성능 확인과 설정 검증이 목적이므로, 최종 통합 테스트 브랜치인 `develop`에는 병합하지 않습니다.
