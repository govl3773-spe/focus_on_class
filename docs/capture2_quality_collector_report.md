# 1-3 Capture2 품질 필터 수집기 보고서

## 결론

`1-3_capture2.ipynb`는 CNN 학습용 얼굴 crop 데이터셋을 만들기 위한 **품질 필터형 수집기**다. 사용자가 `1`, `2`, `3`으로 라벨을 직접 선택하되, MediaPipe 기반 점수가 `CAPTURE_THRESHOLD` 이상일 때만 저장 대기열에 넣어 애매한 샘플 유입을 줄인다.

핵심은 자동 분류가 아니라 **좋은 학습 데이터만 통과시키는 게이트**다.

## 동작 요약

1. 웹캠 프레임을 읽고 좌우 반전한다.
2. MediaPipe로 얼굴, 손, 포즈를 탐지한다.
3. 얼굴 landmark에서 crop bbox와 품질 feature를 계산한다.
4. `Attractive`, `Drowsy`, `Inattentive` 점수를 만든다.
5. 사용자가 누른 클래스 점수가 기준 이상이면 얼굴 crop을 임시 저장한다.
6. `s`를 누르면 이미지와 `metadata.csv`를 함께 저장한다.

## 수집 기준

기본 threshold는 다음 설정값으로 관리한다.

```python
CAPTURE_THRESHOLD = 0.40
CROP_OUTPUT_SIZE = (224, 224)
MIN_FACE_AREA_RATIO = 0.015
```

- `CAPTURE_THRESHOLD`: 선택한 클래스 점수가 이 값 이상일 때만 캡처한다.
- `CROP_OUTPUT_SIZE`: CNN 입력용 얼굴 crop 크기다.
- `MIN_FACE_AREA_RATIO`: 얼굴이 너무 작게 잡힌 프레임을 낮은 품질로 본다.

## 유틸 함수 핵심

### `make_mp_image`

OpenCV/NumPy RGB 프레임을 MediaPipe 입력 포맷으로 바꾼다.

```python
def make_mp_image(frame_rgb: np.ndarray) -> mp.Image:
```

역할은 단순하지만 중요하다. MediaPipe Tasks API는 `mp.Image`를 입력으로 받으므로, 매 프레임마다 이 변환이 탐지 파이프라인의 시작점이 된다.

### `clamp01`

점수를 `0.0 ~ 1.0` 범위로 고정한다.

```python
def clamp01(value: float) -> float:
```

클래스 점수는 여러 feature를 조합해 만들기 때문에 범위를 벗어날 수 있다. 이 함수는 score가 threshold 비교에 안정적으로 쓰이도록 보장한다.

### `face_landmarks_to_bbox`

MediaPipe 얼굴 landmark 전체를 감싸는 bbox를 만들고, 가장자리에 약간의 여백을 더한다.

```python
def face_landmarks_to_bbox(face_landmarks, frame_width: int, frame_height: int):
```

이 함수가 수집 품질의 기준점이다. bbox가 정확해야 얼굴 crop도 정확하고, 얼굴 면적과 중앙 위치 feature도 의미가 생긴다.

핵심 처리:

- landmark의 x, y 좌표 최솟값/최댓값으로 얼굴 영역 계산
- 얼굴 크기의 12%를 padding으로 추가
- 프레임 바깥으로 나가지 않도록 좌표 clamp

### `crop_first_face_bgr`

첫 번째 얼굴 bbox를 기준으로 저장용 crop 이미지를 만든다.

```python
def crop_first_face_bgr(frame_bgr: np.ndarray, face_boxes, output_size=CROP_OUTPUT_SIZE):
```

저장되는 이미지는 화면에 표시되는 주석 프레임이 아니라, 원본 BGR 프레임에서 잘라낸 얼굴 crop이다. 따라서 bbox, landmark, 상태 텍스트가 학습 이미지에 섞이지 않는다.

반환값이 `None`인 경우:

- 얼굴 bbox가 없음
- crop 결과가 비어 있음

### `landmark_distance`

두 얼굴 landmark 사이의 픽셀 거리를 계산한다.

```python
def landmark_distance(face_landmarks, a: int, b: int, width: int, height: int) -> float:
```

현재는 눈 세로/가로 비율을 계산하는 데 사용한다. normalized landmark 좌표를 실제 프레임 크기 기준 거리로 바꿔 비교한다.

### `eye_open_score`

눈이 떠 있을수록 1에 가까운 점수를 만든다.

```python
def eye_open_score(face_landmarks, width: int, height: int) -> float:
```

좌우 눈의 세로 거리와 가로 거리를 비교해 눈 뜸 정도를 추정한다. 이 값은 `Attractive` 점수를 높이고, 반대로 `Drowsy` 점수에서는 눈 감김 신호로 사용된다.

주의할 점:

- 조명, 안경, 얼굴 각도에 민감할 수 있다.
- 완성된 졸림 판정 모델이 아니라 수집 품질 필터다.

### `calculate_class_scores`

수집기에서 가장 중요한 함수다. MediaPipe 결과를 이용해 세 클래스 점수와 metadata feature를 계산한다.

```python
def calculate_class_scores(face_result, face_boxes, frame_shape):
```

계산하는 feature:

- `face_area`: 얼굴 bbox가 프레임에서 차지하는 비율
- `eye_open`: 눈 뜸 정도
- `head_down`: 코 위치 기반 고개 숙임 추정값
- `off_center`: 얼굴이 화면 중심에서 벗어난 정도

클래스별 의도:

- `Attractive`: 얼굴이 충분히 크고, 중앙에 있고, 눈이 떠 있고, 고개가 많이 숙여지지 않은 프레임
- `Drowsy`: 얼굴은 잡히지만 눈 감김과 고개 숙임이 강한 프레임
- `Inattentive`: 얼굴이 중심에서 벗어나거나 자세가 흐트러진 프레임

이 함수의 점수는 모델 예측 확률이 아니다. CNN 학습 전 데이터 품질을 높이기 위한 규칙 기반 신뢰도다.

## 화면 표시 함수

화면 표시 함수들은 수집자의 판단을 돕기 위한 시각화 계층이다.

- `draw_blue_segmentation_overlay`: 사람 영역을 파란색 overlay로 표시
- `draw_face_landmarks`: 얼굴 bbox와 윤곽 landmark 표시
- `draw_hand_landmarks`: 손 landmark 표시
- `draw_pose_landmarks`: 포즈 landmark 표시
- `draw_status`: FPS, 얼굴/손 개수, 클래스별 점수 표시

저장 이미지는 이 시각화 결과가 아니라 원본 프레임 crop이다. 화면 표시는 수집 보조용이고, 학습 데이터에는 들어가지 않는다.

## 저장 함수

`save_pending_images`는 대기열의 crop 이미지를 클래스 폴더에 저장하고, `append_metadata`로 CSV 기록을 남긴다.

저장 구조:

```text
captured_images/
  Attractive/
  Drowsy/
  Inattentive/
  metadata.csv
```

`metadata.csv`에는 이미지 경로, 라벨, 점수, feature가 기록된다. 나중에 잘못 저장된 샘플을 추적하거나 threshold를 조정할 때 기준 자료로 쓸 수 있다.

## 해석 기준

이 수집기는 사용자의 라벨 선택을 대체하지 않는다. 사용자가 라벨을 선택하고, MediaPipe 점수는 해당 라벨이 너무 애매한지 걸러내는 역할을 한다.

즉, 목표는 다음과 같다.

- 잘린 얼굴 제거
- 얼굴이 너무 작은 샘플 제거
- 클래스 기준과 너무 동떨어진 샘플 제거
- CNN에 넣을 crop 크기 통일
- 추후 검수 가능한 metadata 확보

## 조정 포인트

수집 결과가 너무 적으면 `CAPTURE_THRESHOLD`를 낮춘다. 애매한 이미지가 많이 섞이면 `CAPTURE_THRESHOLD`를 올린다.

눈 감김이나 고개 숙임 기준이 체감과 다르면 `eye_open_score`, `head_down` 계산식을 조정한다. 이 두 값은 특히 사람, 카메라 높이, 조명에 따라 흔들릴 수 있다.

## 다음 단계

충분한 데이터를 모은 뒤에는 바로 학습하지 말고 클래스별 샘플을 먼저 검수한다. 그 다음 train/validation/test split을 만들고, 작은 CNN 또는 전이학습 모델로 첫 분류 실험을 진행하는 흐름이 적절하다.
