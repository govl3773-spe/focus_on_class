# 실시간 집중도 분류 모델 선정 보고서

## 1. 개요

본 프로젝트는 웹캠 기반 실시간 수업 집중도 분석 시스템을 구현하는 것을 목표로 한다. 사용자의 얼굴 영상을 입력받아 CNN 기반 이미지 분류 모델을 통해 `Attentive`, `Drowsy`, `LookingAway` 상태를 판단하고, FastAPI 기반 UI에서 실시간으로 결과를 시각화한다.

모델 후보로는 다음 세 가지 CNN 계열 모델을 비교하였다.

- ResNet18
- MobileNetV3 Large
- EfficientNet-B0

초기 검증 데이터셋 기준의 정량 지표에서는 ResNet18이 가장 높은 성능을 보였다. 그러나 FastAPI 기반 실시간 UI에 적용한 결과, 실제 웹캠 환경에서는 MobileNetV3 Large가 가장 안정적인 예측 흐름을 보였다. 따라서 본 보고서에서는 정량 평가 결과와 실제 UI 적용 결과가 다르게 나타난 원인을 분석하고, 최종 모델 선정 근거를 정리한다.

---

## 2. 모델별 정량 평가 결과

| 모델 | Validation Accuracy | Validation F1-score | 추론 시간 | 파라미터 수 | 특징 |
|---|---:|---:|---:|---:|---|
| ResNet18 | 0.8363 | 0.8286 | 0.717 ms | 약 11.18M | 검증 성능 최고 |
| MobileNetV3 Large | 0.8304 | 0.8265 | 0.637 ms | 약 4.21M | 경량성 및 실시간성 우수 |
| EfficientNet-B0 | 0.8070 | 0.8037 | 0.771 ms | 약 4.01M | 파라미터 수는 적지만 성능 낮음 |

정량 평가만 보면 ResNet18이 가장 높은 accuracy와 F1-score를 기록하였다. 그러나 ResNet18과 MobileNetV3 Large의 F1-score 차이는 다음과 같이 매우 작다.

```text
0.8286 - 0.8265 = 0.0021
```

즉, 두 모델의 검증 성능 차이는 약 0.21%p 수준으로, 데이터 분할 방식이나 random seed 변화에 따라 순위가 바뀔 수 있는 근소한 차이라고 볼 수 있다.

---

## 3. FastAPI 실시간 UI 적용 결과

세 모델을 FastAPI 기반 실시간 UI에 번갈아 적용한 결과, 사용자가 체감한 안정성은 다음과 같았다.

```text
MobileNetV3 Large ≥ EfficientNet-B0 >>> ResNet18
```

또한 모델별 초기 실행 시간은 다음과 같이 측정되었다.

| 모델 | 초기 실행 시간 |
|---|---:|
| ResNet18 | 약 27초 |
| MobileNetV3 Large | 약 33초 |
| EfficientNet-B0 | 약 35초 |

초기 실행 시간만 보면 ResNet18이 가장 빠르게 실행되었다. 그러나 이 시간은 모델 로딩 시간만을 의미하지 않는다. 실제 FastAPI 서비스 시작 과정에는 다음 요소들이 함께 포함된다.

- CNN 모델 로드
- MediaPipe FaceLandmarker 로드
- SQLite DB 초기화
- 세션 생성
- OpenCV 카메라 연결
- 첫 프레임 수신
- 브라우저 MJPEG 스트리밍 연결
- 판단 윈도우 누적 시간

따라서 27초, 33초, 35초의 차이를 단순히 모델 무게나 파라미터 수 차이로만 해석하기는 어렵다.

---

## 4. 정량 지표와 실제 UI 결과가 다르게 나타난 이유

### 4.1 검증 데이터와 실제 웹캠 입력의 분포 차이

모델 비교표의 validation 성능은 저장된 이미지 데이터셋을 기준으로 측정된다. 반면 실제 FastAPI UI에서는 웹캠 프레임이 다음 과정을 거쳐 모델에 입력된다.

```text
웹캠 프레임
→ MediaPipe 얼굴 검출
→ 얼굴 bounding box 생성
→ padding 적용
→ 얼굴 crop
→ resize
→ CNN 모델 입력
```

이 과정에서 학습 및 검증 데이터와 실제 입력 데이터 사이에 차이가 발생할 수 있다. 예를 들어 실제 UI 환경에서는 조명, 얼굴 각도, 카메라 거리, 배경, 얼굴 crop 영역, 프레임 흔들림 등이 계속 변한다.

따라서 ResNet18이 정제된 validation 데이터에서는 높은 점수를 보였더라도, 실제 웹캠 환경에서는 입력 분포 변화에 더 민감하게 반응했을 가능성이 있다.

---

### 4.2 ResNet18의 상대적 과적합 가능성

ResNet18은 MobileNetV3 Large보다 파라미터 수가 많다.

| 모델 | 파라미터 수 |
|---|---:|
| ResNet18 | 약 11.18M |
| MobileNetV3 Large | 약 4.21M |
| EfficientNet-B0 | 약 4.01M |

파라미터 수가 많다는 것은 모델의 표현력이 크다는 장점이 있지만, 데이터 수가 충분하지 않거나 클래스 기준이 명확하지 않을 경우 학습 데이터의 특정 특징을 과하게 외울 가능성도 있다.

예를 들어 학습 데이터에서 `Attentive` 클래스는 정면 얼굴이 많고, `LookingAway` 클래스는 측면 얼굴이 많았다면, ResNet18은 얼굴 방향뿐 아니라 조명, 배경, 얼굴 크기 같은 부수적인 특징까지 함께 학습했을 수 있다.

반면 MobileNetV3 Large는 더 가볍고 단순한 구조이기 때문에 실제 웹캠 입력에서 오히려 덜 예민하게 반응하고, 더 안정적인 결과를 보였을 가능성이 있다.

---

### 4.3 학습 전처리와 실시간 UI 전처리의 차이

학습 과정에서는 보통 다음과 같은 전처리가 사용된다.

```python
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(ImageNet mean/std)
```

또는 데이터 증강을 포함하여 다음과 같은 방식이 사용될 수 있다.

```python
RandomResizedCrop(224)
RandomHorizontalFlip()
RandomRotation()
ColorJitter()
Normalize(ImageNet mean/std)
```

그러나 FastAPI 실시간 UI에서는 모델 입력을 만들기 위해 얼굴 crop 이후 이미지를 정사각형으로 resize하고, ImageNet 평균과 표준편차로 정규화한다.

```python
transforms.Resize((image_size, image_size))
transforms.ToTensor()
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

이때 학습 당시의 이미지 비율, crop 방식, resize 방식과 실제 UI의 입력 방식이 다르면 모델별로 성능 저하 정도가 다르게 나타날 수 있다.

특히 ResNet18이 학습 당시의 crop 형태나 얼굴 위치에 더 강하게 의존했다면, 실제 UI에서 MediaPipe 기반 crop 결과가 조금만 달라져도 예측이 흔들릴 수 있다.

---

### 4.4 ResNet18과 MobileNetV3 Large의 검증 성능 차이가 매우 작음

정량 평가 결과에서 ResNet18이 가장 높은 성능을 보였지만, MobileNetV3 Large와의 차이는 매우 작았다.

| 모델 | Validation F1-score |
|---|---:|
| ResNet18 | 0.8286 |
| MobileNetV3 Large | 0.8265 |

두 모델의 F1-score 차이는 0.0021로, 사실상 거의 동급이라고 볼 수 있다.

따라서 “ResNet18이 명확하게 우수하다”기보다는, “검증 데이터 기준으로 ResNet18이 근소하게 높았다”고 해석하는 것이 더 적절하다.

이처럼 정량 지표 차이가 작을 때는 실제 서비스 환경에서의 안정성, 추론 흐름, 예측 확률의 일관성을 함께 고려해야 한다.

---

### 4.5 실시간 시스템에서는 단일 이미지 정확도보다 확률 안정성이 중요함

일반적인 validation 성능은 한 장의 이미지에 대해 정답을 맞혔는지 여부를 평가한다. 그러나 본 프로젝트의 실시간 UI는 단일 프레임 하나로 최종 상태를 결정하지 않는다.

현재 시스템은 다음과 같은 방식으로 동작한다.

```text
15프레임 확률 수집
→ 각 클래스 확률 평균 및 카운트 계산
→ 최종 상태 결정
```

즉, 한 프레임의 예측 결과보다 여러 프레임에서 확률이 얼마나 안정적으로 유지되는지가 더 중요하다.

예를 들어 ResNet18이 단일 이미지 기준으로는 정답을 잘 맞히더라도, 프레임마다 확률이 크게 흔들린다면 15프레임 평균 판단에서는 최종 상태가 불안정하게 바뀔 수 있다.

반면 MobileNetV3 Large는 단일 이미지 기준 성능이 ResNet18보다 아주 조금 낮더라도, 프레임 간 확률 변화가 더 부드럽고 일관적이라면 실제 UI에서는 더 안정적인 모델로 보일 수 있다.

따라서 본 프로젝트에서는 단순 validation accuracy보다 실시간 환경에서의 확률 안정성이 더 중요한 판단 기준이 된다.

---

## 5. 모델 무게와 초기 실행 시간에 대한 해석

모델 무게는 실행 속도에 영향을 준다. 특히 다음 항목에는 모델 구조와 파라미터 수가 영향을 미친다.

- 모델 파일 로딩 시간
- GPU 또는 CPU 메모리 사용량
- 프레임당 CNN 추론 시간
- 장시간 실행 시 처리 안정성
- FPS 유지 능력

그러나 현재 FastAPI UI의 초기 실행 시간은 모델 로딩만으로 결정되지 않는다. 서비스 시작 시 MediaPipe FaceLandmarker, OpenCV 카메라, SQLite DB, MJPEG 스트리밍 등이 함께 동작한다.

따라서 ResNet18이 27초, MobileNetV3 Large가 33초, EfficientNet-B0가 35초가 걸렸다고 해서 ResNet18이 실시간 시스템에 가장 적합하다고 단정할 수는 없다.

특히 본 프로젝트에서는 “몇 초 더 빨리 켜지는가”보다 “켜진 뒤 예측 결과가 얼마나 덜 흔들리는가”가 더 중요하다.

---

## 6. 최종 모델 선정

정량 평가에서는 ResNet18이 가장 높은 validation accuracy와 F1-score를 보였다. 그러나 MobileNetV3 Large와의 F1-score 차이는 0.0021로 매우 작았으며, 실제 FastAPI 기반 실시간 UI에서는 MobileNetV3 Large가 더 안정적인 예측 결과를 보였다.

따라서 본 프로젝트에서는 다음 기준을 종합하여 최종 모델을 선정하였다.

| 기준 | 판단 |
|---|---|
| Validation 성능 | ResNet18이 근소하게 우세 |
| 실제 UI 안정성 | MobileNetV3 Large 우세 |
| 모델 경량성 | MobileNetV3 Large 우세 |
| 실시간 적용성 | MobileNetV3 Large 우세 |
| 예측 흐름의 자연스러움 | MobileNetV3 Large 우세 |

최종적으로 본 프로젝트의 기준 모델은 다음과 같이 선정한다.

```text
최종 모델: MobileNetV3 Large
보조 비교 모델: EfficientNet-B0
정량 지표상 최고 모델: ResNet18
```

---

## 7. 보고서용 결론 문장

정량 평가에서는 ResNet18이 가장 높은 validation accuracy와 F1-score를 보였으나, MobileNetV3 Large와의 F1-score 차이는 약 0.0021로 매우 작았다. 이후 FastAPI 기반 실시간 UI에 세 모델을 동일 조건으로 적용한 결과, MobileNetV3 Large가 실제 웹캠 환경에서 더 안정적인 예측 흐름을 보였다.

이는 검증 데이터와 실시간 웹캠 입력 간의 분포 차이, 얼굴 crop 및 resize 방식 차이, ResNet18의 상대적 과적합 가능성, 그리고 실시간 판단에서 요구되는 프레임 간 확률 안정성 차이 때문으로 해석된다.

따라서 본 프로젝트에서는 단순 validation 성능뿐 아니라 실제 서비스 환경에서의 안정성, 경량성, 실시간 적용 가능성을 함께 고려하여 MobileNetV3 Large를 최종 모델로 선정하였다.

---

## 8. 한 줄 요약

ResNet18은 정제된 검증 데이터에서 가장 높은 점수를 보였지만, 실제 웹캠 기반 실시간 UI에서는 MobileNetV3 Large가 더 안정적으로 동작했기 때문에 최종 모델로 MobileNetV3 Large를 선정하였다.
