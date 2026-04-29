# Git Structure Report

## 보고서 목적

이 문서는 `focus_on_class` 저장소의 현재 Git/worktree 구조와 이번 브랜치 정리 작업 결과를 기록하기 위한 보고서입니다.

## 현재 브랜치 구조

저장소는 메인 디렉터리와 브랜치별 nested worktree 구조로 운영합니다.

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

## 브랜치별 역할

- `0_prototype`
  - 초기 프로토타입 실험 브랜치
  - `develop`에는 포함하지 않음

- `1_capture`
  - MediaPipe 기반 데이터 수집 브랜치
  - `develop`에는 `mp_model/`, `1-3_capture2.ipynb`만 반영

- `2_minicnn`
  - Mini CNN / MobileNet / ResNet 개별 테스트 브랜치
  - `develop`에는 포함하지 않음

- `3_compare`
  - 데이터 분리 및 모델 비교 브랜치
  - `develop`에는 `models/`, `dataset_123_split.ipynb`, `1-7_compare.ipynb`, `1-8_all_model_compare.ipynb`, `2-1_three_model_compare.ipynb`만 반영

- `develop`
  - 최종 테스트용 통합 브랜치
  - capture와 compare 결과 중 필요한 파일만 선별 포함

- `4_threshold`
  - OpenCV 실시간 테스트와 threshold 조정 전용 브랜치
  - 현재는 최소 실행 구성만 유지

## 이번 정리 작업 요약

### 1. 브랜치 README 정리

각 작업 브랜치에 목적과 포함 범위를 설명하는 `README.md`를 정리했습니다.

### 2. `develop` 브랜치 생성

`develop` 브랜치를 새 worktree로 만들고, 최종 테스트에 필요한 파일만 선별 반영했습니다.

포함 항목:

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

### 3. `4_threshold` 브랜치 생성

`develop` 기준으로 `4_threshold`를 생성한 뒤, OpenCV 실시간 테스트만 담당하도록 분리했습니다.

현재 `4_threshold` 포함 항목:

- `3-1_openCV_test.ipynb`
- `models/resnet18_best.pt`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `git_structure_report.md`

## `4_threshold` 브랜치 정리 원칙

- 비교 실험 전체를 들고 가지 않음
- OpenCV 테스트용 노트북만 유지
- 실행 환경 동기화를 위해 `README.md`, `pyproject.toml`, `uv.lock` 유지
- 모델 파일은 `resnet18` 체크포인트만 유지

## worktree 매핑

| 브랜치 | 경로 |
|---|---|
| `main` | `C:\Projects\_active\focus_on_class` |
| `0_prototype` | `C:\Projects\_active\focus_on_class\worktrees\0_prototype` |
| `1_capture` | `C:\Projects\_active\focus_on_class\worktrees\1_capture` |
| `2_minicnn` | `C:\Projects\_active\focus_on_class\worktrees\2_minicnn` |
| `3_compare` | `C:\Projects\_active\focus_on_class\worktrees\3_compare` |
| `develop` | `C:\Projects\_active\focus_on_class\worktrees\develop` |
| `4_threshold` | `C:\Projects\_active\focus_on_class\worktrees\4_threshold` |

## 주요 커밋

| 브랜치 | 최신 커밋 |
|---|---|
| `0_prototype` | `afc9c1a docs: add README for prototype branch` |
| `1_capture` | `5bfb770 docs: add README for capture branch` |
| `2_minicnn` | `1876db6 docs: add README for minicnn branch` |
| `3_compare` | `dcb7620 docs: add README for compare branch` |
| `develop` | `0614ce4 feat: assemble develop test branch` |
| `4_threshold` | `10fbcf2 chore: restore threshold branch sync files` |

## 결론

현재 구조는 실험 브랜치와 최종 테스트 브랜치를 분리하고, `4_threshold`는 OpenCV 테스트 전용 최소 구성으로 유지하는 방향으로 정리되어 있습니다.
