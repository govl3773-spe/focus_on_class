# focus_on_class 구조도 및 흐름도

## 1. 전체 시스템 구조도

```mermaid
flowchart LR
    A[Webcam Input] --> B[OpenCV Frame Capture]
    B --> C[MediaPipe Face Landmarker]
    C --> D[Face Crop Preprocessing]
    D --> E[PyTorch CNN Model]
    E --> F[Class Probability]
    F --> G[Temporal Aggregation]
    G --> H[Focus State Decision]
    H --> I[FastAPI Dashboard]
    H --> J[JSON API]
    H --> K[SQLite Session Log]

    E --> E1[ResNet18]
    E --> E2[MobileNetV3 Large]
    E --> E3[EfficientNet-B0]
```

## 2. 실시간 추론 흐름도

```mermaid
sequenceDiagram
    participant User as User
    participant Cam as Webcam
    participant CV as OpenCV
    participant MP as MediaPipe
    participant Model as PyTorch Model
    participant API as FastAPI
    participant DB as SQLite

    User->>API: Start session
    API->>Cam: Open camera
    Cam->>CV: Send frame
    CV->>MP: Pass frame
    MP->>CV: Return face region
    CV->>Model: Send cropped face image
    Model->>API: Return class probabilities
    API->>API: Aggregate recent predictions
    API->>DB: Save session log
    API->>User: Stream video and focus state
```

## 3. 모델 분류 흐름도

```mermaid
flowchart TD
    A[Face Crop Image] --> B[Resize and Normalize]
    B --> C[CNN Inference]
    C --> D{Prediction Result}
    D -->|Highest probability| E[Attentive]
    D -->|Highest probability| F[Drowsy]
    D -->|Highest probability| G[LookingAway]
    E --> H[Recent Prediction Buffer]
    F --> H
    G --> H
    H --> I[Threshold / Majority Logic]
    I --> J[Final Session State]
```

## 4. 개발 단계 흐름도

```mermaid
flowchart LR
    A[0_prototype<br/>MediaPipe 초기 검증]
    B[1_capture<br/>얼굴 crop 데이터 수집]
    C[2_minicnn<br/>경량 CNN 학습 실험]
    D[3_compare<br/>후보 모델 비교]
    E[4_threshold<br/>실시간 threshold 검증]
    F[5_service<br/>FastAPI 서비스 구현]
    G[develop<br/>최종 실행 산출물 통합]

    A --> B --> C --> D --> E --> F --> G
```

## 5. 브랜치별 산출물 구조도

```mermaid
flowchart TD
    ROOT[focus_on_class]

    ROOT --> P[0_prototype]
    ROOT --> C[1_capture]
    ROOT --> M[2_minicnn]
    ROOT --> CMP[3_compare]
    ROOT --> T[4_threshold]
    ROOT --> S[5_service]
    ROOT --> D[develop]

    P --> P1[MediaPipe / landmark prototype]
    C --> C1[1-3_capture2.ipynb]
    C --> C2[capture quality report]
    M --> M1[training notebooks]
    CMP --> CMP1[model comparison notebooks]
    CMP --> CMP2[model selection report]
    T --> T1[OpenCV threshold validation]
    S --> S1[service/app.py]
    S --> S2[model checkpoints]
    S --> S3[service guide]
    D --> D1[final runnable service]
    D --> D2[docs]
    D --> D3[minimum model assets]
```

## 6. 최종 실행 파일 구조

```text
focus_on_class
├─ README.md
├─ PROJECT_STRUCTURE_FLOW.md
├─ worktrees/
│  ├─ 0_prototype/
│  ├─ 1_capture/
│  ├─ 2_minicnn/
│  ├─ 3_compare/
│  ├─ 4_threshold/
│  ├─ 5_service/
│  │  ├─ service/
│  │  │  ├─ app.py
│  │  │  └─ GUIDE.md
│  │  ├─ models/
│  │  └─ mp_model/
│  └─ develop/
│     ├─ service/
│     │  └─ app.py
│     ├─ models/
│     │  ├─ resnet18_best.pt
│     │  ├─ mobilenet_v3_large_best.pt
│     │  └─ efficientnet_b0_best.pt
│     ├─ mp_model/
│     │  └─ face_landmarker.task
│     ├─ docs/
│     ├─ pyproject.toml
│     └─ uv.lock
└─ docs/
```

## 7. 포트폴리오용 요약 흐름

```mermaid
flowchart LR
    A[데이터 수집] --> B[모델 학습]
    B --> C[모델 비교]
    C --> D[실시간 threshold 검증]
    D --> E[FastAPI 서비스화]
    E --> F[웹캠 기반 집중도 분석]

    A1[OpenCV / face crop] -.-> A
    B1[PyTorch / TorchVision] -.-> B
    C1[scikit-learn / visualization] -.-> C
    D1[OpenCV realtime test] -.-> D
    E1[FastAPI / SQLite / MJPEG] -.-> E
```
