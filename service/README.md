# Focus On Class FastAPI Service

`service/app.py`는 OpenCV 웹캠 추론 흐름을 FastAPI + MJPEG 스트리밍 서비스로 옮긴 실행 파일입니다. 모델은 15프레임 단위로 확률을 모아 최종 상태를 결정하고, 결과를 SQLite DB에 저장합니다.

## 실행

프로젝트 루트(`C:\Projects\focus_on_class`)에서 실행합니다.

```powershell
uv run uvicorn service.app:app --reload --host 127.0.0.1 --port 8000
```

필요 패키지를 별도로 설치해야 하면 다음 명령을 먼저 실행합니다.

```powershell
uv pip install -r service\requirements-service.txt
```

브라우저 주소:

```text
http://127.0.0.1:8000
```

## 주요 API

- `GET /`: 실시간 영상과 상태 패널 UI
- `GET /video_feed`: OpenCV 처리 프레임을 MJPEG로 스트리밍
- `POST /api/start`: 카메라, MediaPipe, 모델을 열고 새 세션 시작
- `POST /api/stop`: 추론 루프 종료, 세션 종료 시간 저장
- `GET /api/status`: 현재 상태, 확률, warning, FPS 반환
- `GET /api/summary`: 현재 또는 최근 세션의 집중도 요약 반환

## 사용 자산

- 모델: `models/{model_name}_best.pt`
- 기본 모델명: `resnet18`
- MediaPipe 얼굴 모델: `mp_model/face_landmarker.task`
- DB 저장 위치: `service/attention_logs.db`

## 상태 처리

서비스는 모델 출력명을 다음 최종 상태명으로 정규화합니다.

| 입력 가능 이름 | 최종 상태 |
|---|---|
| `attentive` | `Attentive` |
| `drowsy` | `Drowsy` |
| `lookingaway`, `looking_away`, `looking away`, `inattentive` | `LookingAway` |

15프레임 안에서 `Attentive` 신호가 충분하면 집중 상태를 보호하고, 그렇지 않으면 `LookingAway`와 `Drowsy` 평균 확률을 비교합니다. 최근 3개 결과가 모두 `Drowsy` 또는 모두 `LookingAway`일 때 warning을 표시합니다.
