# MobileNet 학습 노트북

이 폴더는 `resnet` / `resnet_dataset` 흐름과 같은 방식으로 MobileNet 계열 모델을 학습하고 웹캠 실시간 추론까지 실행하는 노트북을 담고 있습니다.

## 폴더 구조

```text
mobilenet/
  mobilenet.ipynb
  README.md
  checkpoints/

mobilenet_dataset/
  train/
    Attentive/
    Drowsy/
    LookingAway/
  val/
    Attentive/
    Drowsy/
    LookingAway/
  test/
    Attentive/
    Drowsy/
    LookingAway/
```

## 실행 방법

1. Jupyter에서 `mobilenet/mobilenet.ipynb`를 엽니다.
2. 위에서 아래로 셀을 순서대로 실행합니다.
3. 기본 모델은 `mobilenet_v3_small`입니다.
4. 마지막 셀에서 웹캠 실시간 추론을 실행합니다.
5. 모델을 바꾸려면 설정 셀의 `BACKBONE` 값을 변경합니다.

## 지원 모델

- `mobilenet_v2`
- `mobilenet_v3_small`
- `mobilenet_v3_large`

## 데이터 준비 규칙

- 원본 데이터는 `dataset/Attentive`, `dataset/Drowsy`, `dataset/LookingAway` 구조를 사용합니다.
- 노트북 실행 시 `mobilenet_dataset/train`, `mobilenet_dataset/val`, `mobilenet_dataset/test`로 자동 분할됩니다.
- 학습된 최고 성능 모델은 `mobilenet/checkpoints` 아래에 저장됩니다.
