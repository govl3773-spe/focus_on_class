# CNN 모델 성능 비교 및 최종 모델 선정 보고서

## 1. 보고서 목적

본 보고서는 `1-7_compare.ipynb`에서 수행한 CNN 모델 비교 결과를 바탕으로, 실시간 수업 집중도 판단 시스템에 적합한 최종 모델을 선정하기 위해 작성하였다.

본 프로젝트의 목표는 웹캠으로 얼굴 이미지를 입력받아 다음 세 가지 상태를 분류하는 것이다.

- Attentive
- Drowsy
- LookingAway

따라서 단순히 테스트 정확도만 높은 모델을 선택하는 것보다, 실제 Gradio 또는 Streamlit 기반 실시간 감지 환경에서 안정적으로 동작할 수 있는지를 함께 고려해야 한다. 이를 위해 Accuracy, F1-score, Inference Time, Model File Size, Parameters를 종합적으로 비교하였다.

---

## 2. 비교 대상 모델

이번 비교에 사용한 모델은 `models` 폴더에 저장된 학습 완료 모델 8종이다.

| 계열 | 모델 |
|---|---|
| ResNet | ResNet18, ResNet34, ResNet50 |
| MobileNet | MobileNetV3-Large |
| EfficientNet | EfficientNet-B0, EfficientNet-B2, EfficientNet-B3, EfficientNet-V2-S |

모든 모델은 동일한 `dataset/test` 데이터셋을 기준으로 평가하였다. 평가 데이터는 학습 과정에 사용하지 않은 test split이므로, 모델의 최종 일반화 성능을 비교하는 기준으로 사용하였다.

---

## 3. 비교 기준

모델 비교에는 다음 지표를 사용하였다.

| 지표 | 의미 |
|---|---|
| Accuracy | 전체 test 이미지 중 맞게 예측한 비율 |
| Macro F1-score | 클래스별 F1-score를 평균낸 값 |
| Inference Time | 이미지 1장당 추론 시간 |
| Model File Size | 저장된 모델 파일의 크기 |
| Parameters | 모델이 가진 학습 파라미터 수 |

이 프로젝트에서는 실시간 웹캠 추론이 중요하기 때문에 Accuracy와 F1-score뿐 아니라 Inference Time도 매우 중요한 기준이다. Training Time은 모델을 학습할 때 한 번 발생하는 비용이지만, Inference Time은 사용자가 웹캠을 켜고 있는 동안 매 프레임마다 반복적으로 발생하기 때문이다.

---

## 4. 전체 비교 결과

`1-7_compare.ipynb` 실행 결과는 다음과 같다.

| Model | Accuracy | Macro F1 | Inference Time (ms/image) | Model Size (MB) | Parameters (M) | Best Val F1 |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.9940 | 0.9941 | 2.727 | 15.59 | 4.01 | 0.9886 |
| ResNet18 | 0.9940 | 0.9939 | 1.456 | 42.72 | 11.18 | 0.9886 |
| ResNet50 | 0.9940 | 0.9939 | 2.680 | 90.00 | 23.51 | 0.9827 |
| EfficientNet-V2-S | 0.9880 | 0.9883 | 2.602 | 77.86 | 20.18 | 0.9886 |
| EfficientNet-B3 | 0.9880 | 0.9881 | 2.436 | 41.36 | 10.70 | 0.9886 |
| EfficientNet-B2 | 0.9880 | 0.9880 | 2.163 | 29.83 | 7.71 | 0.9943 |
| MobileNetV3-Large | 0.9880 | 0.9880 | 1.373 | 16.25 | 4.21 | 0.9943 |
| ResNet34 | 0.9820 | 0.9822 | 1.862 | 81.34 | 21.29 | 0.9886 |

테스트 성능만 보면 EfficientNet-B0, ResNet18, ResNet50이 Accuracy 0.9940으로 가장 높은 그룹에 속한다. 다만 이 세 모델 사이에서도 추론 시간과 모델 크기에는 차이가 있다.

---

## 5. Accuracy 및 F1-score 분석

Accuracy 기준으로 가장 높은 모델은 EfficientNet-B0, ResNet18, ResNet50이다. 세 모델 모두 test accuracy가 0.9940으로 동일하게 나타났다.

Macro F1-score 기준으로는 EfficientNet-B0이 0.9941로 가장 높다. ResNet18과 ResNet50도 0.9939로 거의 같은 수준이므로, 순수 분류 성능만 보면 세 모델의 차이는 매우 작다.

반면 MobileNetV3-Large는 test accuracy 0.9880, macro F1-score 0.9880으로 최고 성능 그룹보다 약간 낮다. 그러나 이 차이는 크지 않으며, 실시간 시스템에서는 추론 속도와 모델 크기 측면의 장점이 더 중요하게 작용할 수 있다.

---

## 6. Inference Time 분석

실시간 웹캠 기반 집중도 판단에서는 Inference Time이 매우 중요하다. 모델이 아무리 정확해도 추론이 느리면 화면 반응이 늦어지고, 사용자는 상태 변화가 부자연스럽다고 느낄 수 있다.

가장 빠른 모델은 MobileNetV3-Large로, 이미지 1장당 1.373ms가 걸렸다. 그 다음은 ResNet18로 1.456ms이다.

| 속도 순위 | Model | Inference Time (ms/image) |
|---:|---|---:|
| 1 | MobileNetV3-Large | 1.373 |
| 2 | ResNet18 | 1.456 |
| 3 | ResNet34 | 1.862 |
| 4 | EfficientNet-B2 | 2.163 |
| 5 | EfficientNet-B3 | 2.436 |
| 6 | EfficientNet-V2-S | 2.602 |
| 7 | ResNet50 | 2.680 |
| 8 | EfficientNet-B0 | 2.727 |

EfficientNet-B0은 가장 높은 Macro F1-score를 보였지만, 추론 시간은 2.727ms로 비교 대상 중 가장 느린 편이다. 반대로 MobileNetV3-Large는 성능이 약간 낮지만 가장 빠르기 때문에 실시간 적용성 측면에서 강점이 있다.

---

## 7. 모델 크기 및 파라미터 분석

모델 파일 크기와 파라미터 수는 배포 편의성, 로딩 속도, 메모리 사용량에 영향을 준다.

모델 파일 크기 기준으로 가장 작은 모델은 EfficientNet-B0이며 15.59MB이다. MobileNetV3-Large도 16.25MB로 매우 작다.

파라미터 수 기준으로는 EfficientNet-B0이 4.01M으로 가장 적고, MobileNetV3-Large가 4.21M으로 거의 비슷하다.

| Model | Model Size (MB) | Parameters (M) |
|---|---:|---:|
| EfficientNet-B0 | 15.59 | 4.01 |
| MobileNetV3-Large | 16.25 | 4.21 |
| EfficientNet-B2 | 29.83 | 7.71 |
| EfficientNet-B3 | 41.36 | 10.70 |
| ResNet18 | 42.72 | 11.18 |
| EfficientNet-V2-S | 77.86 | 20.18 |
| ResNet34 | 81.34 | 21.29 |
| ResNet50 | 90.00 | 23.51 |

ResNet50은 Accuracy는 높지만 파일 크기와 파라미터 수가 가장 큰 편이다. 따라서 실시간 웹캠 서비스나 가벼운 데모 환경에서는 부담이 될 수 있다.

---

## 8. 모델별 장단점

### 8.1 EfficientNet-B0

장점:
- 가장 높은 Macro F1-score를 기록하였다.
- 모델 파일 크기와 파라미터 수가 가장 작다.
- 정확도와 경량성의 균형이 좋다.

단점:
- 이번 측정에서는 inference time이 가장 느린 편이었다.
- 실시간 웹캠 환경에서는 MobileNetV3-Large나 ResNet18보다 반응성이 낮을 수 있다.

### 8.2 ResNet18

장점:
- Accuracy와 F1-score가 최고 성능 그룹에 속한다.
- MobileNetV3-Large 다음으로 빠르다.
- 구조가 단순하고 안정적이다.

단점:
- 파일 크기와 파라미터 수가 EfficientNet-B0, MobileNetV3-Large보다 크다.
- 경량 배포 측면에서는 MobileNet 계열보다 불리하다.

### 8.3 ResNet50

장점:
- Accuracy와 F1-score가 최고 성능 그룹에 속한다.
- 깊은 구조를 통해 안정적인 표현력을 가질 수 있다.

단점:
- 파일 크기와 파라미터 수가 가장 크다.
- 추론 시간이 비교적 느리다.
- 성능 차이에 비해 실시간 시스템에서 얻는 이점이 크지 않다.

### 8.4 MobileNetV3-Large

장점:
- 가장 빠른 inference time을 기록하였다.
- 모델 파일 크기와 파라미터 수가 작다.
- 실시간 웹캠 기반 서비스에 적합하다.
- Accuracy와 F1-score도 충분히 높은 수준이다.

단점:
- 최고 성능 그룹보다 test accuracy와 macro F1-score가 약간 낮다.
- 아주 작은 성능 차이까지 중요하게 보는 경우에는 EfficientNet-B0 또는 ResNet18보다 불리할 수 있다.

### 8.5 EfficientNet-B2, B3, V2-S

장점:
- EfficientNet 계열의 확장 모델로 일정 수준 이상의 성능을 보였다.
- Validation F1-score가 높은 모델도 있다.

단점:
- 테스트 성능이 EfficientNet-B0보다 높지 않았다.
- 모델 크기와 추론 시간이 증가했지만, 그에 비례하는 성능 향상은 확인되지 않았다.
- 실시간 감지 목적에서는 B0 또는 MobileNetV3-Large보다 선택 우선순위가 낮다.

### 8.6 ResNet34

장점:
- ResNet18보다 깊은 구조를 가진 안정적인 CNN 모델이다.

단점:
- 이번 test 결과에서는 Accuracy와 F1-score가 가장 낮았다.
- ResNet18보다 느리고 파일 크기도 크므로, 현재 결과만 보면 선택 근거가 약하다.

---

## 9. Accuracy만으로 모델을 선택하면 안 되는 이유

이번 결과에서 EfficientNet-B0, ResNet18, ResNet50은 Accuracy가 모두 0.9940으로 동일하다. 따라서 Accuracy만 보면 세 모델 중 어떤 모델을 선택해도 비슷해 보인다.

그러나 실제 실시간 웹캠 시스템에서는 다음 요소가 함께 중요하다.

- 사용자가 움직였을 때 예측 결과가 얼마나 빠르게 반응하는가
- 웹캠 화면이 끊기지 않고 자연스럽게 표시되는가
- 모델 로딩과 실행이 부담스럽지 않은가
- Gradio 또는 Streamlit 환경에서 안정적으로 유지되는가
- 추후 배포 또는 공유가 쉬운가

따라서 최종 모델은 Accuracy뿐 아니라 Inference Time, Model Size, Parameters를 함께 고려하여 선택해야 한다.

---

## 10. 최종 모델 추천

### 추천 1순위: MobileNetV3-Large

본 프로젝트의 최종 목적이 실시간 웹캠 기반 집중도 판단이라면 MobileNetV3-Large를 가장 추천한다.

추천 이유:

- Inference Time이 1.373ms/image로 가장 빠르다.
- 모델 파일 크기가 16.25MB로 작다.
- 파라미터 수가 4.21M으로 가볍다.
- Accuracy 0.9880, Macro F1 0.9880으로 실사용에 충분히 높은 성능을 보인다.
- Gradio 또는 Streamlit 기반 실시간 데모에서 반응성이 좋을 가능성이 높다.

MobileNetV3-Large는 최고 Accuracy 모델은 아니지만, 실시간 시스템에서 중요한 속도와 경량성을 가장 잘 만족한다. 따라서 실제 웹캠 데모 또는 사용자 체감 품질을 우선한다면 MobileNetV3-Large가 적합하다.

### 대안 추천: EfficientNet-B0

정확도를 가장 우선한다면 EfficientNet-B0도 좋은 선택이다.

추천 이유:

- Macro F1-score가 0.9941로 가장 높다.
- 모델 크기가 15.59MB로 가장 작다.
- 파라미터 수가 4.01M으로 가장 적다.

다만 이번 측정에서는 Inference Time이 2.727ms/image로 가장 느린 편이었다. 따라서 실시간 반응성보다 분류 성능을 더 중요하게 볼 때 선택하는 것이 좋다.

### 대안 추천: ResNet18

성능과 속도의 균형을 원한다면 ResNet18도 좋은 후보이다.

추천 이유:

- Accuracy 0.9940, Macro F1 0.9939로 최고 성능 그룹에 속한다.
- Inference Time이 1.456ms/image로 빠르다.
- 구조가 단순하고 안정적이다.

단점은 모델 파일 크기가 42.72MB로 MobileNetV3-Large나 EfficientNet-B0보다 크다는 점이다.

---

## 11. 결론

이번 CNN 모델 비교 결과, 테스트 정확도만 보면 EfficientNet-B0, ResNet18, ResNet50이 가장 높은 성능을 보였다. 그러나 본 프로젝트는 단순 이미지 분류가 아니라 웹캠 기반 실시간 집중도 판단 시스템이므로, 추론 속도와 모델 경량성도 중요한 기준이다.

종합적으로 판단하면 최종 추천 모델은 MobileNetV3-Large이다. MobileNetV3-Large는 Accuracy와 F1-score가 최고 모델보다 약간 낮지만, 가장 빠른 inference time과 작은 모델 크기를 가지고 있어 실시간 데모 환경에 가장 적합하다.

다만 최종 배포 목적에 따라 선택은 달라질 수 있다.

| 목적 | 추천 모델 |
|---|---|
| 실시간 반응성 우선 | MobileNetV3-Large |
| 최고 F1-score 우선 | EfficientNet-B0 |
| 정확도와 속도 균형 | ResNet18 |
| 큰 모델도 괜찮고 정확도 우선 | ResNet50 |

따라서 본 프로젝트에서는 `MobileNetV3-Large`를 최종 실시간 감지용 모델로 우선 적용하고, 이후 실제 웹캠 환경에서 Attentive, Drowsy, LookingAway 상태 변화가 자연스럽게 표시되는지 추가 검증하는 것이 적절하다.

---

## 12. 향후 개선 방향

향후에는 다음 작업을 진행하면 모델과 서비스 품질을 더 높일 수 있다.

- 실시간 웹캠에서 오분류되는 장면을 추가 수집하여 재학습
- Attentive 클래스가 덜 나오는 경우 class weight 조정 실험
- smoothing window와 confidence threshold 튜닝
- confusion matrix 기반으로 자주 헷갈리는 클래스 분석
- Gradio 또는 Streamlit 앱에 최종 모델 자동 로딩 적용
- ONNX 또는 TorchScript 변환을 통한 추론 최적화
- 실제 사용 환경에서 FPS와 예측 안정성 측정

이 과정을 거치면 단순히 test set에서 높은 성능을 보이는 모델이 아니라, 실제 웹캠 환경에서도 안정적으로 동작하는 집중도 판단 모델로 발전시킬 수 있다.
