# Git Structure Report

## 1. 보고서 목적

이 문서는 `focus_on_class` 저장소의 현재 Git 브랜치/worktree 구조와 이번 정리 작업 결과를 한 번에 확인하기 위한 보고서입니다.  
특히 `develop` 브랜치 통합 구성과 `4_threshold` 브랜치 분리 목적을 명확하게 기록합니다.

## 2. 현재 저장소 구조

현재 저장소는 메인 작업 디렉터리와 브랜치별 nested worktree 구조로 운영합니다.

```text
C:\Projects\_active\focus_on_class
├─ main
└─ worktrees
   ├─ 0_prototype
   ├─ 1_capture
   ├─ 2_minicnn
   ├─ 3_compare
   ├─ develop
   └─ 4_threshold
```

## 3. 브랜치별 역할

### `main`

- 저장소 기준 루트
- 현재는 통합 테스트 결과물을 직접 쌓는 브랜치가 아니라 worktree 운영의 기준점 역할

### `0_prototype`

- 초기 프로토타입 실험 브랜치
- MediaPipe 기반 초기 아이디어 검증용
- `develop`에는 포함하지 않음

### `1_capture`

- 데이터 수집 및 MediaPipe capture 작업 브랜치
- `develop`에는 아래 항목만 선별 반영
  - `mp_model/`
  - `1-3_capture2.ipynb`

### `2_minicnn`

- Mini CNN / MobileNet / ResNet 개별 테스트 브랜치
- 모델별 실험 브랜치이므로 `develop`에는 포함하지 않음

### `3_compare`

- 데이터 분리 및 모델 비교 실험 브랜치
- `develop`에는 아래 항목만 선별 반영
  - `models/`
  - `dataset_123_split.ipynb`
  - `1-7_compare.ipynb`
  - `1-8_all_model_compare.ipynb`
  - `2-1_three_model_compare.ipynb`

### `develop`

- 최종 테스트용 통합 브랜치
- `1_capture`, `3_compare`에서 필요한 파일만 선별 구성
- 최신 `pyproject.toml`, `uv.lock` 포함

### `4_threshold`

- OpenCV 실시간 테스트와 threshold 조정 전용 브랜치
- `develop` 기준으로 새로 분리 생성
- 비교 노트북과 실시간 테스트 노트북의 역할을 분리하기 위한 브랜치

## 4. worktree 매핑

| 브랜치 | 경로 |
|---|---|
| `main` | `C:\Projects\_active\focus_on_class` |
| `0_prototype` | `C:\Projects\_active\focus_on_class\worktrees\0_prototype` |
| `1_capture` | `C:\Projects\_active\focus_on_class\worktrees\1_capture` |
| `2_minicnn` | `C:\Projects\_active\focus_on_class\worktrees\2_minicnn` |
| `3_compare` | `C:\Projects\_active\focus_on_class\worktrees\3_compare` |
| `develop` | `C:\Projects\_active\focus_on_class\worktrees\develop` |
| `4_threshold` | `C:\Projects\_active\focus_on_class\worktrees\4_threshold` |

## 5. 이번 정리 작업 내용

### 5-1. 브랜치 README 정리

각 작업 브랜치에 목적과 포함 범위를 설명하는 `README.md`를 정리했습니다.

- `0_prototype`: 프로토타입 브랜치 설명 추가
- `1_capture`: capture 자산과 develop 반영 대상 설명 추가
- `2_minicnn`: 개별 모델 테스트 브랜치 설명 추가
- `3_compare`: 모델 비교 노트북과 `models/` 역할 설명 추가

### 5-2. `develop` 브랜치 생성

`develop` 브랜치를 새 worktree로 생성하고, 최종 테스트에 필요한 파일만 선별 반영했습니다.

포함 파일:

- `pyproject.toml`
- `uv.lock`
- `mp_model/`
- `1-3_capture2.ipynb`
- `models/`
- `dataset_123_split.ipynb`
- `1-7_compare.ipynb`
- `1-8_all_model_compare.ipynb`
- `2-1_three_model_compare.ipynb`
- `README.md`

제외 원칙:

- `0_prototype` 전체 제외
- `2_minicnn` 전체 제외
- `1_capture`, `3_compare`는 지정 파일만 선별 반영
- 캐시/임시 파일은 반영하지 않음

### 5-3. `4_threshold` 브랜치 생성

`4_threshold` 브랜치는 `develop`을 기준으로 별도 worktree로 생성했습니다.

추가 작업:

- `3_compare`의 OpenCV 테스트 노트북 내용을 분리
- `3-1_openCV_test.ipynb` 생성
- threshold 조정 작업은 이 브랜치에서만 진행하도록 분리

## 6. `develop`과 `4_threshold`의 차이

### `develop`

- 최종 테스트 기준 브랜치
- capture + compare 결과를 통합한 기준선
- 재현 가능한 실행 순서와 테스트 자산 보관 목적

### `4_threshold`

- 실시간 OpenCV 테스트 전용 브랜치
- threshold 조정과 예측 안정화 실험 전용
- 비교 실험 결과를 유지한 채, 실시간 테스트 파라미터만 별도 조정

## 7. 현재 기준 주요 커밋

| 브랜치 | 최신 커밋 |
|---|---|
| `0_prototype` | `afc9c1a docs: add README for prototype branch` |
| `1_capture` | `5bfb770 docs: add README for capture branch` |
| `2_minicnn` | `1876db6 docs: add README for minicnn branch` |
| `3_compare` | `dcb7620 docs: add README for compare branch` |
| `develop` | `0614ce4 feat: assemble develop test branch` |
| `4_threshold` | `39b03fa chore: create threshold branch and separate opencv test notebook` |

## 8. 현재 `4_threshold` 브랜치 포함 항목

```text
models/
mp_model/
1-3_capture2.ipynb
1-7_compare.ipynb
1-8_all_model_compare.ipynb
2-1_three_model_compare.ipynb
3-1_openCV_test.ipynb
dataset_123_split.ipynb
pyproject.toml
uv.lock
README.md
git_structure_report.md
```

## 9. 결론

현재 구조는 다음 원칙으로 정리되어 있습니다.

1. 실험 브랜치와 최종 테스트 브랜치를 분리한다.
2. `develop`에는 필요한 파일만 선별 반영한다.
3. OpenCV 실시간 테스트와 threshold 조정은 `4_threshold`에서 별도로 진행한다.
4. worktree 기반으로 브랜치별 작업 공간을 분리해 충돌을 줄인다.
