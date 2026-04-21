# focus_on_class용 MediaPipe 변수명 정리표

이 문서는 기존 `mp_model`, `mp_data`, `5-3_face_landmark.ipynb`, `5-5_pose_landmark.ipynb` 구조를  
새 프로젝트인 `focus_on_class`로 옮길 때 사용할 **변수명 정리 가이드**이다.

---

## 1. 유지할 폴더 구조

아래 폴더 형태는 그대로 유지한다.

- `mp_model/`
- `mp_data/`
- 얼굴 / 자세 관련 notebook 파일

---

## 2. 바뀌는 점

- `mp_data/`는 일단 **빈 폴더 구조만 유지**한다.
- 새로운 데이터를 나중에 다시 수집할 예정이므로,  
  기존 CSV / XML / 이미지 파일은 새 프로젝트로 복사하지 않는다.
- 이번 작업에서는 **폴더명 유지 + notebook 내부 변수명 정리**만 수행한다.

---

## 3. 권장 네이밍 규칙

도메인별 접두어를 명확하게 나눈다.

- 얼굴 관련 코드: `face_...`
- 자세 관련 코드: `pose_...`
- 공통 MediaPipe 보조 요소: `mp_...`

프레임 관련 변수명은 아래처럼 통일한다.

- OpenCV 원본 프레임: `frame_bgr`
- RGB 변환 프레임: `frame_rgb`
- 최종 출력 프레임: `annotated_frame`

시간값은 아래처럼 통일한다.

- `timestamp_ms`

추론 결과 객체는 아래처럼 통일한다.

- 얼굴 결과: `face_result`
- 자세 결과: `pose_result`

landmarker 객체는 아래처럼 통일한다.

- 얼굴: `face_landmarker`
- 자세: `pose_landmarker`

옵션 객체는 아래처럼 통일한다.

- 얼굴: `face_options`
- 자세: `pose_options`

---

## 4. 얼굴 노트북 변수명 변경표

기존 이름 -> 권장 이름

- `model_path` -> `face_model_path`
- `options` -> `face_options`
- `landmarker` -> `face_landmarker`
- `cap` -> `camera_capture`
- `ret` -> `is_readable`
- `frame` -> `frame_bgr`
- `mp_image` -> `face_mp_image` 또는 `input_image`
- `timestamp` -> `timestamp_ms`
- `face_landmarker_result` -> `face_result`
- `annotated_image` -> `face_annotated_image`
- `draw_landmarks_on_image` -> `draw_face_landmarks_on_image`
- `face_landmarks_list` -> `face_landmarks_batch`
- `face_landmarks` -> `single_face_landmarks`
- `idx` -> `face_idx`
- `fig` -> `blendshape_fig`
- `ax` -> `blendshape_ax`
- `bar` -> `blendshape_bar`
- `face_blendshapes_category` -> `blendshape_category`
- `face_blendshapes_names` -> `blendshape_names`
- `face_blendshapes_scores` -> `blendshape_scores`
- `face_blendshapes_ranks` -> `blendshape_ranks`
- `score` -> `blendshape_score`
- `patch` -> `bar_patch`
- `x` -> `bar_x`
- `key` -> `pressed_key`

---

## 5. 자세 노트북 변수명 변경표

기존 이름 -> 권장 이름

- `model_path` -> `pose_model_path`
- `options` -> `pose_options`
- `landmarker` -> `pose_landmarker`
- `cap` -> `camera_capture`
- `ret` -> `is_readable`
- `frame` -> `frame_bgr`
- `rgb_frame` -> `frame_rgb`
- `mp_image` -> `pose_mp_image` 또는 `input_image`
- `timestamp` / `ts` -> `timestamp_ms`
- `pose_landmarker_result` -> `pose_result`
- `annotated_image` -> `pose_annotated_image`
- `draw_landmarks_on_image` -> `draw_pose_landmarks_on_image`
- `draw_segmentation_on_image` -> `draw_pose_segmentation_on_image`
- `pose_landmarks_list` -> `pose_landmarks_batch`
- `pose_landmarks` -> `single_pose_landmarks`
- `pose_landmark_style` -> `pose_landmark_draw_style`
- `pose_connection_style` -> `pose_connection_draw_style`
- `overlay` -> `segmentation_overlay`
- `segmentation_mask` -> `segmentation_mask_2d`
- `rgb` -> `frame_rgb`
- `annotated` -> `annotated_frame`
- `key` -> `pressed_key`

---

## 6. 얼굴 + 자세 통합 노트북에서의 권장 이름

얼굴 / 자세를 하나의 viewer notebook으로 합칠 경우 아래 이름을 권장한다.

그대로 유지해도 되는 클래스명:

- `BaseOptions`
- `FaceLandmarker`
- `PoseLandmarker`
- `FaceLandmarkerOptions`
- `PoseLandmarkerOptions`

단, 실행 모드 이름은 한 가지 스타일로만 통일한다.

- `VisionRunningMode` 또는 `RunningMode` 중 하나만 사용

통합 notebook에서 권장하는 변수명:

- `face_options`
- `pose_options`
- `face_result`
- `pose_result`
- `face_annotated_frame`
- `pose_annotated_frame`
- `combined_annotated_frame`

---

## 7. 새 프로젝트에서 다시 연결해야 할 경로

`focus_on_class` 프로젝트로 옮기면서 아래 경로를 새 기준으로 다시 맞춘다.

- `mp_model/face_landmarker.task`
- `mp_model/pose_landmarker_full.task`

주의:
- 기존 `mp_data` 내부 파일에 의존하지 않는다.
- 새 프로젝트에서는 `mp_data/`를 비워 둔 상태로 시작한다.

---

## 8. 최소 마이그레이션 체크리스트

- `mp_model/` 폴더 유지
- 현재 사용하는 `.task` 파일만 복사
- `mp_data/`는 빈 폴더로 유지
- notebook 내부 변수명을 위 기준에 맞게 정리
- 처음에는 얼굴 / 자세 로직을 분리해서 유지
- 필요할 때 나중에 통합 notebook으로 병합

---

## 9. 정리

이번 마이그레이션의 핵심은 아래와 같다.

- **폴더 구조는 유지**
- **기존 데이터는 가져오지 않음**
- **노트북 변수명만 새 프로젝트 스타일로 통일**
- **얼굴 / 자세 기능은 처음엔 분리**
- **이후 `focus_on_class` 목적에 맞게 탐지 → crop → 데이터 수집 → CNN 학습으로 확장**
