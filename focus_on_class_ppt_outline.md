# Gemma PPT Prompt

아래 조건에 맞춰 `focus_on_class` 프로젝트 발표용 PPT 내용을 만들어줘.

조건:
- 총 10장으로 구성
- 각 장은 반드시 `---` 로 구분
- 발표 시간은 10분 이내
- 청중은 개발자/기술 면접관/프로젝트 리뷰어라고 가정
- 말투는 발표 슬라이드에 맞게 짧고 명확하게
- 너무 마케팅 문구처럼 쓰지 말고, 개발 과정과 설계 의도가 보이게 작성
- 각 장마다 다음 형식으로 작성
  - `# 슬라이드 제목`
  - `핵심 메시지: ...`
  - `본문 불릿:` 아래에 3~5개
  - `발표자 멘트:` 2~4문장
- 프로젝트 흐름은 반드시 `1_capture -> 2_minicnn -> 3_compare -> 4_threshold -> develop`
- 왜 브랜치를 나눴는지, 왜 `develop`에는 일부만 남겼는지, 왜 `4_threshold`가 따로 필요한지 설명할 것
- 숫자를 억지로 만들지 말고, 확인되지 않은 성능 수치는 쓰지 말 것
- 마지막 장에는 한계와 다음 단계까지 포함할 것
- 전체 톤은 "모델 하나를 자랑하는 발표"가 아니라 "실험을 분리하고 통합한 개발 과정 발표"로 유지할 것

디자인 조건:
- main theme color: `#13501B`
- sub/background theme color: `#FEFDE8`
- 전반적인 분위기는 깔끔한 개발 발표 스타일
- 흰색 배경 기본형보다, 따뜻한 아이보리 배경 위에 진한 녹색 포인트가 들어가는 스타일

프로젝트 핵심 정보:
- 프로젝트명: `focus_on_class`
- 목표: 얼굴 기반으로 집중 상태를 분류하는 실험형 비전 프로젝트
- 주요 상태 예시: `Attentive`, `Drowsy`, `LookingAway`
- 저장소는 Git worktree 구조로 운영
- 브랜치 역할
  - `1_capture`: MediaPipe 기반 얼굴 검출, 얼굴 crop 수집, 입력 데이터 준비
  - `2_minicnn`: Mini CNN, MobileNet, ResNet 계열 실험 및 전이학습 구조 확인
  - `3_compare`: 데이터 분리, 여러 모델 비교, 후보 모델 압축
  - `4_threshold`: OpenCV 실시간 테스트, threshold 조정, 판정 안정화
  - `develop`: 최종 테스트용 통합 브랜치
- `1_capture`의 핵심 자산
  - `mp_model/`
  - `1-3_capture2.ipynb`
- `2_minicnn`의 핵심 자산
  - `mobilenet/mobilenet.ipynb`
  - `resnet/resnet_test.ipynb`
- `3_compare`의 핵심 자산
  - `dataset_123_split.ipynb`
  - `1-8_all_model_compare.ipynb`
  - `1-7_compare.ipynb`
  - `2-1_three_model_compare.ipynb`
- `4_threshold`의 핵심 자산
  - `3-1_openCV_test.ipynb`
- `develop`에 포함한 것
  - `mp_model/`
  - `1-3_capture2.ipynb`
  - `models/`
  - `dataset_123_split.ipynb`
  - `1-7_compare.ipynb`
  - `1-8_all_model_compare.ipynb`
  - `2-1_three_model_compare.ipynb`
- 개발 관점 핵심 포인트
  - 데이터 수집과 모델 실험을 분리했다
  - 실험용 브랜치와 최종 테스트 브랜치를 분리했다
  - 오프라인 비교 성능과 실시간 사용자 경험은 다르기 때문에 `4_threshold` 단계가 필요했다
  - `develop`은 모든 파일을 모은 브랜치가 아니라, 최종 테스트에 필요한 최소 통합본이다

슬라이드 흐름은 아래 순서를 따라줘:

---

# 1. 프로젝트 소개

포함 내용:
- 프로젝트 한 줄 소개
- 어떤 문제를 해결하려는지
- 이번 발표를 개발 과정 중심으로 설명한다는 점

---

# 2. 문제 정의와 목표

포함 내용:
- 집중 상태 분류 문제 정의
- 왜 데이터 수집, 모델 비교, 실시간 테스트가 모두 필요한지
- 최종 목표를 파이프라인 관점에서 설명

---

# 3. 전체 개발 구조

포함 내용:
- Git worktree 기반 브랜치 구조
- `1_capture -> 2_minicnn -> 3_compare -> 4_threshold -> develop` 흐름
- 브랜치를 기능별로 분리한 이유

---

# 4. 1_capture

포함 내용:
- MediaPipe 기반 수집 단계의 역할
- 얼굴 crop과 입력 데이터 준비의 중요성
- 왜 수집 단계가 먼저 안정화되어야 하는지

---

# 5. 2_minicnn

포함 내용:
- Mini CNN / MobileNet / ResNet 실험 목적
- 전이학습 구조를 확인한 이유
- 이 단계가 최종본이 아니라 탐색 단계였다는 점

---

# 6. 3_compare

포함 내용:
- 데이터 분리와 비교 기준 통일
- 여러 모델을 같은 기준으로 비교한 이유
- 후보 모델을 압축하는 과정

---

# 7. 4_threshold

포함 내용:
- OpenCV 실시간 테스트를 따로 분리한 이유
- 오프라인 성능과 실시간 체감 성능의 차이
- threshold 및 판정 안정화의 필요성

---

# 8. develop

포함 내용:
- 왜 최종 통합 브랜치가 필요한지
- 왜 일부 실험 파일만 선택적으로 포함했는지
- 최소 통합본 관점에서 `develop`을 설명

---

# 9. 개발자 관점의 핵심 정리

포함 내용:
- 이 프로젝트에서 중요한 것은 모델 하나가 아니라 실험 구조였다는 점
- 재현성, 책임 분리, 유지보수 관점의 장점
- 발표에서 강조할 핵심 메시지 정리

---

# 10. 한계와 다음 단계

포함 내용:
- 현재 구조의 한계
- 데이터 다양성, 실시간 환경 대응, threshold 일반화 문제
- 다음 개선 방향

