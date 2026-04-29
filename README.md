# focus_on_class 5_service

## 개요

`5_service` 브랜치는 최종 집중도 분류 모델을 FastAPI 서비스로 묶은 실행 브랜치입니다.
웹캠 입력을 받아 MediaPipe로 얼굴 영역을 찾고, 세 개의 후보 모델 중 선택한 모델로
실시간 상태를 분류한 뒤 MJPEG 화면과 세션 분석 결과를 함께 제공합니다.

## 포함 파일

- `service/app.py`: FastAPI 서버와 실시간 추론 로직
- `models/efficientnet_b0_best.pt`
- `models/mobilenet_v3_large_best.pt`
- `models/resnet18_best.pt`
- `mp_model/`: MediaPipe task 파일
- `model_selection_report.md`: 최종 모델 선정 근거 정리
- `pyproject.toml`, `uv.lock`: 실행 환경 정의

## 실행 방법

프로젝트 루트에서 아래 순서로 실행합니다.

```powershell
uv sync
uv run uvicorn service.app:app --reload --host 127.0.0.1 --port 8000
```

브라우저 주소:

```text
http://127.0.0.1:8000
```

## 서비스 흐름

1. 웹캠 프레임을 읽습니다.
2. MediaPipe Face Landmarker로 얼굴 위치를 찾습니다.
3. 얼굴 crop을 세 모델 중 선택된 기본 모델에 넣어 확률을 계산합니다.
4. 15프레임 단위로 확률을 모아 `Attentive`, `Drowsy`, `LookingAway` 상태를 결정합니다.
5. 세션 로그를 `service/attention_logs.db`에 저장합니다.
6. `/api/summary`, `/api/analysis`에서 요약과 분석 결과를 제공합니다.

## 기본 설정

- 기본 실시간 모델: `efficientnet_b0`
- 얼굴 task 파일: `mp_model/face_landmarker.task`
- 세션 DB: `service/attention_logs.db`

## 주요 API

- `GET /`: 실시간 대시보드
- `GET /video_feed`: MJPEG 비디오 스트림
- `POST /api/start`: 추론 세션 시작
- `POST /api/stop`: 추론 세션 종료
- `GET /api/status`: 현재 상태 조회
- `GET /api/summary`: 세션 요약 조회
- `GET /api/analysis`: 세션 분석 조회
