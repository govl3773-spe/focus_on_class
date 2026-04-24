# 1-3 Capture2 품질 수집 가이드

`1-3_capture2.ipynb`의 품질 게이트(quality gate) 기반 수집 로직과 최신 점수 조정 사항을 정리한 문서입니다.

## 목적

- `Attractive(집중)`와 `Inattentive(집중 저하)`가 동시에 높게 나오는 겹침을 줄입니다.
- `Inattentive`는 화면 이탈/고개 회전/기울기 신호가 강할 때 명확하게 올라가도록 합니다.
- 수집 단계에서 클래스 경계가 더 분리된 샘플을 저장하도록 quality gate를 조정합니다.

## 기본 설정

```python
CAPTURE_THRESHOLD = 0.60
CROP_OUTPUT_SIZE = (224, 224)
MIN_FACE_AREA_RATIO = 0.015
```

- `CAPTURE_THRESHOLD`: 선택한 클래스 점수가 이 값 이상일 때만 pending 큐에 적재
- `CROP_OUTPUT_SIZE`: 저장되는 얼굴 crop 크기
- `MIN_FACE_AREA_RATIO`: 얼굴이 너무 작게 잡힌 프레임을 품질 낮음으로 처리하기 위한 기준

## 핵심 feature

- `face_area`: 얼굴 bbox가 프레임에서 차지하는 비율
- `eye_open`: 눈 뜸 정도
- `head_down`: 고개 숙임 추정치
- `off_center`: 얼굴 중심의 화면 중앙 이탈 정도
- `face_yaw`: 좌우 회전(yaw) 정도
- `head_tilt`: 좌우 기울기(roll) 정도
- `frontal`: 정면성 점수 (`1 - max(face_yaw, head_tilt)`)

## 최신 조정 사항

### 1) Inattentive 증거치 강화

기존 단순 가중합 대신, 이탈 신호를 묶는 `inattentive_evidence`를 중심으로 계산합니다.

```python
posture_break = clamp01(max(off_center, face_yaw, head_tilt))
inattentive_evidence = clamp01(
    0.50 * posture_break
    + 0.20 * off_center
    + 0.20 * face_yaw
    + 0.10 * head_down
)
```

### 2) Attractive 상호 억제 게이트 추가

`Inattentive` 증거가 강한 프레임에서는 `Attractive`가 동시에 높아지지 않도록 게이트를 적용합니다.

```python
attractive_base = clamp01(
    0.30 * quality
    + 0.30 * centered
    + 0.20 * eye_open
    + 0.20 * frontal
)
attractive_gate = clamp01(1.0 - 0.75 * inattentive_evidence)
scores["Attractive"] = clamp01(attractive_base * attractive_gate)
```

## 현재 클래스 점수식

```python
scores["Attractive"] = clamp01(attractive_base * attractive_gate)

scores["Drowsy"] = clamp01(
    0.20 * quality
    + 0.50 * eye_closed
    + 0.25 * head_down
    + 0.05 * head_tilt
)

scores["Inattentive"] = clamp01(
    0.10 * quality
    + 0.85 * inattentive_evidence
    + 0.05 * (1.0 - frontal)
)
```

## 기대 효과

- `Inattentive` 샘플: 옆보기/기울기/중앙 이탈 프레임이 더 우선 저장
- `Attractive` 샘플: 정면/중앙/안정 자세 프레임 중심으로 정제
- 클래스 간 경계가 선명해져 학습 데이터 품질 개선

## 튜닝 가이드

- `Inattentive`가 약하면: `attractive_gate` 계수 `0.75`를 `0.80~0.90`으로 상향
- `Attractive`가 너무 적게 저장되면: `CAPTURE_THRESHOLD`를 `0.55`로 완화
- `Inattentive`가 과검출이면: `inattentive_evidence`의 `posture_break` 가중치(`0.50`)를 소폭 하향
