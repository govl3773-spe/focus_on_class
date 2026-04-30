# Focus On Class Service Guide

이 문서는 `service/app.py` 실행 방법과 내부 판단 로직을 정리한 가이드입니다.

## 1. 실행 위치

서비스는 프로젝트 루트에서 실행합니다.

```powershell
cd C:\Projects\focus_on_class
uv pip install -r service\requirements-service.txt
uv run uvicorn service.app:app --reload --host 127.0.0.1 --port 8000
```

브라우저 접속 주소:

```text
http://127.0.0.1:8000
```

## 2. 필요한 폴더 구조

`service` 폴더만 교체해서 공유하는 경우에도, 프로젝트 루트에는 아래 리소스가 있어야 합니다.

```text
focus_on_class/
  service/
    app.py
    GUIDE.md
    README.md
    requirements-service.txt
    attention_logs.db        # 자동 생성 가능
  models/
    resnet18_best.pt         # 기본 실시간 모델
  mp_model/
    face_landmarker.task     # MediaPipe 얼굴 검출 모델
```

DB 파일인 `service/attention_logs.db`는 없어도 됩니다. 실행 중 세션 로그가 저장되면서 자동 생성됩니다.

## 3. 주요 의존성

`service/requirements-service.txt` 기준:

```text
fastapi
uvicorn[standard]
opencv-python
mediapipe
numpy
pillow
matplotlib
torch
torchvision
```

## 4. 기본 설정값

`app.py` 상단 상수에서 설정합니다.

| 항목 | 현재 값 | 의미 |
|---|---:|---|
| `REALTIME_MODEL_NAME` | `resnet18` | 사용할 모델 이름. `models/resnet18_best.pt` 필요 |
| `CAMERA_INDEX` | `0` | 기본 웹캠 번호 |
| `FRAME_WIDTH` | `960` | 카메라 입력 너비 |
| `FRAME_HEIGHT` | `720` | 카메라 입력 높이 |
| `FACE_MODEL_PATH` | `mp_model/face_landmarker.task` | MediaPipe face landmarker 파일 |
| `FACE_PADDING_RATIO` | `0.25` | 얼굴 crop 주변 여백 |

카메라가 안 열리면 `CAMERA_INDEX`를 `1`, `2` 등으로 바꿔봅니다.

## 5. 판단 단위

모델은 매 프레임 추론하지만, 최종 상태는 일정 구간을 모아서 결정합니다.

| 항목 | 현재 값 | 의미 |
|---|---:|---|
| `SAMPLE_INTERVAL_SEC` | `0.2` | 0.2초마다 1개 샘플 저장 |
| `WINDOW_SIZE` | `15` | 15개 샘플마다 최종 상태 결정 |
| `MIN_FRAMES_FOR_DECISION` | `5` | 최소 판단 프레임 수 |

즉 최종 상태는 대략 `0.2초 * 15 = 3초` 단위로 DB에 저장됩니다.

## 6. 최종 상태 보정 규칙

최종 상태는 `decide_state()`에서 결정합니다.

### 6.1 기본 상태

모델 클래스는 아래 상태로 정규화됩니다.

```text
Attentive
Drowsy
LookingAway
Unknown
```

`NoFace`, `Uncertain`, `Unknown` 샘플이 너무 많으면 최종 상태는 `Unknown`으로 처리됩니다.

```python
NO_FACE_ALLOWED_COUNT = 5
UNCERTAIN_ALLOWED_COUNT = 5
```

15개 샘플 중 얼굴 없음 또는 불확실 샘플이 각각 5개를 초과하면 `Unknown`입니다.

### 6.2 Attentive 보호 규칙

```python
ATTENTIVE_FRAME_THRESHOLD = 0.35
ATTENTIVE_MIN_COUNT = 5
```

15개 샘플 중 `Attentive` 확률이 `0.35` 이상인 샘플이 5개 이상이면 최종 상태를 `Attentive`로 둡니다.

### 6.3 Drowsy / LookingAway 비교

Attentive 보호 규칙이 걸리지 않으면, `LookingAway` 평균과 `Drowsy` 평균을 비교합니다.

```text
LookingAway 평균 >= Drowsy 평균 -> LookingAway
그 외 -> Drowsy
```

### 6.4 LookingAway 보너스 규칙

고개가 내려가거나 정면이 아닌 상태가 Drowsy와 헷갈리는 경우를 줄이기 위해 LookingAway에 판단용 보너스를 줍니다.

```python
LOW_ATTENTIVE_AVG_THRESHOLD = 0.15
DROWSY_LOOKINGAWAY_MARGIN_THRESHOLD = 0.15
LOOKINGAWAY_BONUS = 0.08
```

조건:

```text
attentive_avg <= 0.15
그리고 abs(drowsy_avg - lookingaway_avg) <= 0.15
```

위 조건이면 판단용 `LookingAway` 평균에 `+0.08`을 더합니다.


주의:

- DB에 저장되는 `lookingaway_avg` 원본 값은 바꾸지 않습니다.
- 최종 판단 비교에만 `adjusted_lookingaway_avg`를 사용합니다.
- 디버그 정보에 `lookingaway_bonus_applied`, `lookingaway_bonus`, `adjusted_lookingaway_avg`가 남습니다.

## 7. 실시간 화면 구성

메인 화면은 다음 요소로 구성됩니다.

- 카메라 MJPEG 스트림
- 카메라 오른쪽 세로형 집중 비율 게이지
- 우측 실시간 상태 패널
- 종료 후 세션 분석 이미지

집중 비율 게이지는 가중 점수가 아니라 아래 식입니다.

```text
집중 비율 = Attentive / (Attentive + Drowsy + LookingAway) * 100
```

`Unknown`은 얼굴 없음/판단 불가 상태라 비율 계산에서 제외합니다.

## 8. 세션 분석 값

종료 버튼을 누르면 DB에 저장된 현재 세션을 읽어 분석 이미지를 생성합니다.

### 요약 이미지

엔드포인트:

```text
GET /api/analysis/summary-plot
```

포함 내용:

- 총 분석 시간
- 집중 비율
- 가중 집중 점수
- 상태 전환 횟수

### 기본 분석 이미지

엔드포인트:

```text
GET /api/analysis/plot
```

포함 내용:

- 상태 변화 타임라인
- 상태별 누적 시간
- 구간별 평균 확률

### 가중 집중 점수

가중 집중 점수는 단순 집중 비율이 아니라 비집중 상태에 감점을 주는 점수입니다.

```text
가중 집중 점수 =
집중 비율
- 졸음 비율 * 120
- 시선 이탈 비율 * 80
```

결과는 `0~100` 사이로 제한합니다.

## 9. 주요 API

| API | 설명 |
|---|---|
| `GET /` | 웹 UI |
| `GET /video_feed` | MJPEG 카메라 스트림 |
| `POST /api/start` | 실시간 추론 세션 시작 |
| `POST /api/stop` | 세션 종료 및 종료 시간 저장 |
| `GET /api/status` | 실시간 상태, 확률, FPS, 집중 비율 |
| `GET /api/summary` | 세션 요약 JSON |
| `GET /api/analysis` | 세션 상세 분석 JSON |
| `GET /api/analysis/summary-plot` | 요약 PNG 이미지 |
| `GET /api/analysis/plot` | 기본 분석 PNG 이미지 |

## 10. 구동 흐름

1. 사용자가 `시작` 버튼을 누릅니다.
2. `FocusService`가 별도 스레드에서 실행됩니다.
3. OpenCV가 카메라 프레임을 읽습니다.
4. MediaPipe FaceLandmarker가 얼굴 위치를 찾습니다.
5. 얼굴 영역을 crop합니다.
6. PyTorch 모델이 `Attentive`, `Drowsy`, `LookingAway` 확률을 예측합니다.
7. 0.2초마다 확률 샘플을 하나 저장합니다.
8. 15개 샘플이 모이면 `decide_state()`로 최종 상태를 결정합니다.
9. 최종 상태와 디버그 값을 SQLite DB에 저장합니다.
10. 사용자가 `종료` 버튼을 누르면 세션 종료 시간이 저장됩니다.
11. 종료 후 분석 PNG 이미지가 생성되어 화면에 표시됩니다.

## 11. 새로고침 관련 처리

페이지 새로고침 시 무거운 분석 이미지 생성은 자동으로 실행하지 않습니다.

- 새로고침 시: 상태 갱신과 카메라 스트림만 다시 연결
- 시작 시: 기존 분석 이미지를 비움
- 종료 시: 분석 이미지 생성

또한 `/video_feed` 연결이 끊기면 MJPEG generator가 빠져나오도록 처리되어 있습니다.

## 12. MediaPipe timestamp 처리

`detect_for_video()`는 timestamp가 계속 증가해야 합니다.

그래서 세션을 새로 시작할 때마다 `FaceLandmarker`를 새로 생성합니다. 이렇게 하면 새 세션에서 timestamp가 다시 `0ms`부터 시작해도 이전 세션의 timestamp 상태와 충돌하지 않습니다.

