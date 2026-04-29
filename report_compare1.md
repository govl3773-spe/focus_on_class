# CNN 모델 성능 비교 및 최종 후보 선정 보고서

## 1. 목적

본 보고서는 `1-7_compare.ipynb`와 `model_compare_results.csv`의 결과를 기준으로, 웹캠 기반 수업 집중도 판단 서비스에 사용할 CNN 모델 후보를 비교하기 위해 작성했습니다.

분류 대상은 다음 3개 상태입니다.

- `Attentive`: 집중
- `Drowsy`: 졸음
- `LookingAway`: 시선 이탈

## 2. 비교 대상

| 계열 | 모델 |
|---|---|
| ResNet | ResNet18, ResNet34, ResNet50 |
| MobileNet | MobileNetV3-Large |
| EfficientNet | EfficientNet-B0, EfficientNet-B2, EfficientNet-B3, EfficientNet-V2-S |

## 3. 주요 결과

| 모델 | Test Accuracy | Macro F1 | 추론 시간(ms/image) | 파일 크기(MB) | 파라미터(M) |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.9940 | 0.9941 | 2.73 | 15.59 | 4.01 |
| ResNet18 | 0.9940 | 0.9939 | 1.46 | 42.72 | 11.18 |
| ResNet50 | 0.9940 | 0.9939 | 2.68 | 90.00 | 23.51 |
| EfficientNet-V2-S | 0.9880 | 0.9883 | 2.60 | 77.86 | 20.18 |
| EfficientNet-B3 | 0.9880 | 0.9881 | 2.44 | 41.36 | 10.70 |
| EfficientNet-B2 | 0.9880 | 0.9880 | 2.16 | 29.83 | 7.71 |
| MobileNetV3-Large | 0.9880 | 0.9880 | 1.37 | 16.25 | 4.21 |
| ResNet34 | 0.9820 | 0.9822 | 1.86 | 81.34 | 21.29 |

## 4. 해석

정확도만 보면 EfficientNet-B0, ResNet18, ResNet50이 같은 test accuracy를 보였습니다. 그러나 실시간 웹캠 서비스에서는 정확도만으로 모델을 고르기 어렵습니다. 매 프레임 또는 짧은 주기마다 추론해야 하므로 추론 시간, 모델 파일 크기, 운영 안정성도 함께 봐야 합니다.

ResNet18은 최고 정확도 그룹에 포함되면서 추론 시간이 1.46ms/image로 빠릅니다. 파일 크기는 MobileNetV3-Large보다 크지만, 현재 서비스 기본 모델로 쓰기에 충분히 단순하고 안정적인 후보입니다.

MobileNetV3-Large는 정확도가 0.9880으로 최고 그룹보다 약간 낮지만, 추론 시간이 1.37ms/image로 가장 빠르고 파일 크기도 16.25MB로 작습니다. 배포 환경이 가볍거나 CPU 성능이 제한적이면 강한 후보입니다.

EfficientNet-B0는 정확도와 파일 크기의 균형이 좋지만, 현재 측정 기준에서는 ResNet18과 MobileNetV3-Large보다 추론 시간이 느립니다.

## 5. 결론

현재 서비스 기본 모델은 `resnet18_best.pt`를 유지하는 것이 합리적입니다. 정확도 최상위 그룹에 있고 추론 속도도 충분히 빠르기 때문입니다.

추가 배포 최적화가 필요하면 `mobilenet_v3_large_best.pt`를 2순위 후보로 두고 실제 웹캠 환경에서 FPS, 지연, 오탐 패턴을 비교하는 것이 좋습니다.

## 6. 추가 검증 항목

- 실제 수업 조명, 거리, 각도 변화가 포함된 데이터로 재평가
- `Drowsy`와 `LookingAway`의 혼동 사례 분석
- 15프레임 후처리 threshold 조정 전후 비교
- CPU 환경에서 FPS와 지연 시간 측정
- TorchScript 또는 ONNX 변환 후 추론 속도 재측정
