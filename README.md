# focus_on_class

웹캠 기반 수업 집중도 판별 시스템입니다. OpenCV로 실시간 프레임을 받고, MediaPipe로 얼굴 영역을 검출한 뒤, PyTorch CNN 모델이 사용자의 상태를 `Attentive`, `Drowsy`, `LookingAway`로 분류합니다. 최종 결과는 FastAPI 서비스에서 MJPEG 스트리밍, 세션 요약, 상태 분석 API로 제공합니다.

## Tech Stack

| 영역 | 기술 |
|---|---|
| Language | Python 3.12 |
| Deep Learning | PyTorch, TorchVision |
| Model Candidates | ResNet18, MobileNetV3 Large, EfficientNet-B0 |
| Computer Vision | OpenCV, MediaPipe Face Landmarker |
| Image Processing | Pillow, TorchVision transforms |
| API Server | FastAPI |
| Realtime Streaming | MJPEG `StreamingResponse` |
| Session Storage | SQLite |
| Data Analysis | scikit-learn, Matplotlib, Seaborn |
| Notebook Workflow | Jupyter, ipykernel |
| Environment | uv, `pyproject.toml`, `uv.lock` |
| GPU Package Source | PyTorch CUDA 12.6 index |

## Architecture

```text
Webcam
  -> OpenCV frame capture
  -> MediaPipe face detection
  -> Face crop preprocessing
  -> PyTorch CNN inference
  -> Temporal aggregation
  -> FastAPI dashboard/API
  -> SQLite session log
```

서비스의 핵심은 프레임 단위 예측을 그대로 확정하지 않고, 일정 구간의 예측을 모아 세션 상태로 해석하는 것입니다. 실시간 화면에서는 현재 예측과 스트리밍 영상을 보여주고, API에서는 누적 집중도 요약과 분석 결과를 제공합니다.

## Key Features

- 웹캠 기반 실시간 집중도 추론
- MediaPipe Face Landmarker 기반 얼굴 검출
- ResNet18, MobileNetV3 Large, EfficientNet-B0 후보 모델 비교
- `Attentive`, `Drowsy`, `LookingAway` 3-class 분류
- threshold와 연속 프레임 기반 후처리
- FastAPI 기반 실시간 대시보드와 JSON API
- SQLite 기반 세션 로그 저장
- 데이터 수집, 모델 비교, 서비스화를 분리한 branch/worktree 실험 구조

## Model Pipeline

1. OpenCV가 웹캠 프레임을 읽습니다.
2. MediaPipe가 얼굴 landmark와 얼굴 영역을 찾습니다.
3. 얼굴 영역을 crop하고 모델 입력 크기에 맞게 전처리합니다.
4. 선택된 CNN 모델이 세 클래스의 확률을 계산합니다.
5. 최근 예측을 모아 상태를 안정화합니다.
6. FastAPI가 영상 스트림, 현재 상태, 세션 요약을 제공합니다.

분류 클래스는 다음과 같습니다.

| 클래스 | 의미 |
|---|---|
| `Attentive` | 화면을 보고 집중하는 상태 |
| `Drowsy` | 졸림, 눈 감김, 집중 저하에 가까운 상태 |
| `LookingAway` | 시선 또는 얼굴 방향이 화면 밖으로 벗어난 상태 |

## Final Service

최종 서비스 기준 파일은 `develop` 또는 `5_service` 브랜치의 산출물을 기준으로 합니다.

```text
service/app.py
models/resnet18_best.pt
models/mobilenet_v3_large_best.pt
models/efficientnet_b0_best.pt
mp_model/face_landmarker.task
pyproject.toml
uv.lock
docs/GUIDE.md
```

실행 예시는 다음과 같습니다.

```powershell
uv sync
uv run uvicorn service.app:app --reload --host 127.0.0.1 --port 8000
```

브라우저에서 접속합니다.

```text
http://127.0.0.1:8000
```

## API Surface

`service/app.py`는 실시간 화면과 세션 분석을 함께 제공합니다.

| Endpoint | 역할 |
|---|---|
| `GET /` | 실시간 대시보드 HTML |
| `GET /video_feed` | MJPEG 영상 스트림 |
| `POST /api/start` | 집중도 추론 세션 시작 |
| `POST /api/stop` | 세션 종료 |
| `GET /api/status` | 현재 추론 상태 조회 |
| `GET /api/summary` | 세션 요약 조회 |
| `GET /api/analysis` | 세션 분석 결과 조회 |

## Development Flow

프로젝트는 실험 단계를 branch/worktree로 분리해서 진행했습니다.

```text
0_prototype -> 1_capture -> 2_minicnn -> 3_compare -> 4_threshold -> 5_service -> develop
```

| 단계 | 목적 | 산출물 |
|---|---|---|
| `0_prototype` | MediaPipe와 landmark 기반 초기 검증 | 초기 노트북, 탐색 문서 |
| `1_capture` | 얼굴 crop 데이터 수집과 라벨 품질 조정 | 수집 노트북, 품질 보고서 |
| `2_minicnn` | 경량 CNN과 전이학습 실험 | 학습 노트북, checkpoint |
| `3_compare` | 후보 모델 성능 비교 | 비교 노트북, 모델 선정 근거 |
| `4_threshold` | 실시간 추론 threshold 조정 | OpenCV 검증 노트북 |
| `5_service` | 최종 서비스 구현 | FastAPI 앱, 모델, 실행 가이드 |
| `develop` | 검증된 최종 산출물 통합 | 제출/시연용 최소 실행 구성 |

## Technical Highlights

- 실시간 CV 파이프라인을 노트북 실험에서 FastAPI 서비스로 전환했습니다.
- 모델 비교 결과를 바탕으로 서비스 후보 checkpoint를 좁히고, 최종 실행 파일만 `develop`에 남겼습니다.
- 프레임 단위 흔들림을 줄이기 위해 최근 예측을 누적해 상태를 판단하는 구조를 적용했습니다.
- MediaPipe 얼굴 검출과 TorchVision 모델 추론을 하나의 서버 루프 안에서 연결했습니다.
- 세션 로그를 SQLite에 저장해 실시간 상태뿐 아니라 종료 후 요약과 분석도 가능하게 했습니다.
- 데이터 수집, 모델 실험, threshold 검증, 서비스화를 branch 단위로 분리해 실험 산출물과 최종 산출물이 섞이지 않도록 관리했습니다.

## Documents

최종 흐름을 설명하는 문서는 `docs/`에 정리되어 있습니다.

- `docs/capture2_quality_collector_report.md`: 데이터 수집과 라벨 품질 기준
- `docs/report_compare1.md`: 후보 CNN 모델 비교 결과
- `docs/model_selection_report.md`: 최종 서비스 모델 선정 근거
- `docs/GUIDE.md`: 서비스 실행 및 동작 설명

## Repository Policy

- `develop`은 전체 실험 파일을 모으는 브랜치가 아니라, 실행 가능한 최종 산출물만 모으는 브랜치입니다.
- 대용량 모델 파일은 서비스에 필요한 checkpoint만 유지합니다.
- 원본 데이터와 수집 산출물은 삭제나 이동보다 복사 기반으로 다룹니다.
- 브랜치별 README는 해당 단계의 목적과 포함/제외 기준을 설명하는 문서로 유지합니다.
