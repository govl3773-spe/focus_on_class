# MediaPipe 웹캠 프로토타입 작업 정리

작성일: 2026-04-21

이 문서는 `focus_on_class` 프로젝트에서 오늘 진행한 MediaPipe 기반 웹캠 탐지 프로토타입 검토 내용을 정리한 기록이다.
현재 단계는 분류 모델 학습이 아니라, **웹캠 기반 탐지 및 전처리 파이프라인 확인 단계**에 초점을 둔다.

## 1. 이번 단계의 목표

- 웹캠에서 사람을 안정적으로 탐지한다.
- 얼굴 위치를 bbox 또는 랜드마크 형태로 확인한다.
- 손 랜드마크를 함께 표시한다.
- 사람 영역은 블루 오버레이로 시각화한다.
- 좌우 반전된 거울 모드로 자연스럽게 보이게 한다.
- 이후 얼굴 crop 저장 단계로 확장 가능한 구조를 유지한다.

## 2. 이번 단계에서 하지 않는 것

- `attentive / drowsy / inattentive` 분류 구현
- CNN 학습 코드 작성
- KNN, SVM 같은 전통 ML 분류 코드 작성
- 상태 분류를 위한 특징 추출 파이프라인 확정

즉, 지금은 **탐지와 시각화만** 다룬다.

## 3. 확인한 파일

- 메인 노트북: `1-1_us_proto.ipynb`
- 기존 참고 노트북: `1-0_prototype.ipynb`
- 모델 파일:
  - `mp_model/face_landmarker.task`
  - `mp_model/hand_landmarker.task`
  - `mp_model/pose_landmarker_full.task`

## 4. 오늘 확인한 핵심 내용

### 4.1 얼굴 탐지

- `FaceLandmarker`를 사용해 얼굴 위치를 확인하는 방향이 맞다.
- 현재 단계에서는 얼굴 분류가 아니라 **얼굴 bbox 탐지**가 목적이다.
- 얼굴 랜드마크를 bbox로 변환하면 나중에 얼굴 crop 저장으로 확장하기 쉽다.

### 4.2 손 탐지

- `HandLandmarker`가 별도 모델로 연결되어 있다.
- 손 랜드마크는 화면에 함께 표시하면 된다.
- 현재는 손 정보로 분류하지 않는다.

### 4.3 사람 영역 세그멘테이션

- `PoseLandmarker`는 세그멘테이션 마스크 확인용으로 사용 가능하다.
- 사람 영역을 파란색 반투명 오버레이로 시각화하는 방향이 적절하다.
- 이 단계에서는 포즈 분류가 아니라, **사람이 어디에 있는지 확인하는 용도**다.

### 4.4 좌우 반전

- 웹캠 프레임은 `cv2.flip(frame_bgr, 1)`로 거울 모드처럼 보이게 처리하는 것이 맞다.

## 5. 노트북 검토 결과

### 5.1 Pose가 안 보인 이유

`1-1_us_proto.ipynb`에서 `PoseLandmarker`는 생성되어 있었지만,
메인 루프에서 **포즈 랜드마크를 실제로 그리는 함수가 호출되지 않았다.**

즉, Pose는 없던 것이 아니라 **출력 단계에서 빠져 있었다.**

### 5.2 중복 호출 문제

메인 루프에 얼굴 bbox 그리기 함수가 중복으로 들어가 있었다.

문제 패턴:

```python
annotated_frame_rgb, face_boxes = draw_face_landmarks(annotated_frame_rgb, face_result)
annotated_frame_rgb, face_boxes = draw_face_landmarks(annotated_frame_rgb, face_result)
```

이 부분은 한 번만 호출해야 한다.

### 5.3 `KeyError: (0, 1)` 원인

이 에러는 포즈 그리기 함수에서 잘못된 스타일 객체를 연결선 스타일로 넣어서 발생했다.

잘못된 형태:

```python
connection_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
```

이 함수는 랜드마크 스타일용 dict이지, 연결선 스타일용 dict가 아니다.

연결선은 다음처럼 단일 `DrawingSpec`을 사용해야 한다.

```python
connection_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
```

## 6. 현재 권장 구조

노트북 셀은 아래처럼 유지하는 것이 가장 단순하다.

### 셀 1. import

- `cv2`
- `time`
- `numpy`
- `mediapipe`

### 셀 2. 모델/옵션 설정

- `FaceLandmarker`
- `HandLandmarker`
- `PoseLandmarker`
- 카메라 해상도, 색상, 모델 경로 설정

### 셀 3. 유틸 함수

- `make_mp_image()`
- `draw_blue_segmentation_overlay()`
- `face_landmarks_to_bbox()`
- `draw_face_landmarks()`
- `draw_hand_landmarks()`
- `draw_pose_landmarks()`
- `draw_status()`

### 셀 4. 메인 실행 루프

- 웹캠 열기
- 좌우 반전
- BGR -> RGB 변환
- MediaPipe 입력 변환
- 얼굴/손/세그멘테이션 결과 표시
- `q` 입력 시 종료
- 카메라 해제 및 창 닫기

## 7. 다음에 바로 고칠 항목

1. `draw_pose_landmarks()`를 메인 루프에서 실제로 호출한다.
2. 얼굴 bbox 함수 중복 호출을 제거한다.
3. `connection_drawing_spec`는 pose 랜드마크 스타일이 아니라 `DrawingSpec`으로 바꾼다.
4. 필요하면 얼굴 crop 저장 함수를 별도 셀로 분리한다.

## 8. 요약

- 지금 단계는 분류가 아니라 **탐지/전처리 프로토타입**이다.
- 얼굴은 `FaceLandmarker` + bbox로 확인한다.
- 손은 `HandLandmarker`로 표시한다.
- 사람 영역은 `PoseLandmarker` 세그멘테이션으로 블루 오버레이를 적용한다.
- `KeyError: (0, 1)`는 pose 연결선 스타일 지정 방식이 잘못되어 발생했다.
- Pose가 안 보인 것은 함수 호출 누락과 중복 호출 문제였다.

