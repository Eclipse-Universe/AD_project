# AD Project — Chemical Process Anomaly Detection

화학 공정(Tennessee Eastman Process 기반) 센서 데이터에서 이상(anomaly)을 탐지하는 프로젝트.
Train은 정상 데이터만 포함되어 있어 **Novelty Detection** 문제로 접근한다. 평가지표는 F1-score.

자세한 대회/데이터 설명은 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) 참고.

## 작업 방식 (Working Agreement)

- 구현은 직접 작성한다. 방향 제시·개념 설명·코드 리뷰·실험 기록을 가이드 역할로 진행한다.
- 모델/피처/파라미터 변경을 제안할 때는 반드시 **이유(가설)**를 먼저 제시하고 합의한 뒤 진행한다.
- 무조건적인 동의 대신, 트레이드오프를 검토해 합리적 방안을 논의한다.
- 리더보드 제출 결과는 매번 [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)에
  변경사항 → 가설 → 점수 → 분석 → 다음 계획 형태로 기록한다.

## 문서

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — 대회/데이터 개요, baseline 코드 흐름, 주의사항
- [docs/CONCEPTS.md](docs/CONCEPTS.md) — 프로젝트에 필요한 사전 개념과 baseline 코드 매핑
- [docs/SRC_DESIGN.md](docs/SRC_DESIGN.md) — `src/` 각 파일의 책임과 설계 이유 (구현은 직접 작성)
- [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) — 실험/제출 기록
- [docs/QA_LOG.md](docs/QA_LOG.md) — 진행하며 나온 "왜?" 질문과 답 (포트폴리오/PPT용 사고 과정 기록)
- [docs/MENTORING_QUESTIONS.md](docs/MENTORING_QUESTIONS.md) — 확신 없이 경험적으로 판단한 지점들, 멘토링용 질문 리스트

## 구조

```
AD_project/
├── baseline_code/
│   ├── baseline.ipynb       # 제공된 baseline (EDA 기록용으로 보존, 신규 실험은 src/에서 진행)
│   └── requirements.txt
├── data/
│   ├── README.md            # 데이터 shape/주의사항 (대용량 csv는 .gitignore 처리)
│   └── sample_submission.csv
├── docs/
│   ├── CONCEPTS.md
│   └── EXPERIMENT_LOG.md
├── src/                      # 모듈화된 파이프라인 (예정, 구현 진행하며 채울 예정)
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   └── infer.py
├── PROJECT_CONTEXT.md
└── .gitignore
```

`src/`는 baseline의 EDA 인사이트와 라벨 변환(-1/1 → 0/1) 로직 등을 참고하되, 실험마다 코드 변경이
git diff로 명확히 추적되도록 스크립트 단위로 재구성한다. baseline.ipynb는 수정하지 않고 EDA 기록
용도로 그대로 둔다.

## 환경 구성

```bash
pip install -r baseline_code/requirements.txt
```

## 데이터

`data/train.csv`, `data/test.csv`는 용량 문제로 저장소에 포함하지 않는다 (`data/README.md` 참고).
