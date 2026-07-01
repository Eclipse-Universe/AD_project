# Chemical Process Anomaly Detection

Tennessee Eastman Process(TEP) 기반 화학 공정 센서 데이터에서 이상을 탐지하는 프로젝트.
Train은 정상 데이터만 포함되어 있는 Novelty Detection 문제이며, 평가지표는 F1-score.

대회/데이터 상세 설명 → [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)

## 문서

| 파일 | 내용 |
|---|---|
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | 대회 개요, 데이터 구조, baseline 코드 흐름 |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | 프로젝트에 필요한 개념 정리 |
| [docs/SRC_DESIGN.md](docs/SRC_DESIGN.md) | `src/` 설계 근거 |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | 실험 및 제출 기록 |
| [docs/QA_LOG.md](docs/QA_LOG.md) | 실험 중 나온 질문과 답 |
| [docs/MENTORING_QUESTIONS.md](docs/MENTORING_QUESTIONS.md) | 멘토링용 질문 리스트 |

## 구조

```
AD_project/
├── baseline_code/
│   ├── baseline.ipynb        # 대회 제공 baseline (수정 없이 보존)
│   └── requirements.txt
├── data/                     # csv 파일은 .gitignore 처리 (data/README.md 참고)
│   └── sample_submission.csv
├── docs/
│   ├── CONCEPTS.md
│   ├── EXPERIMENT_LOG.md
│   ├── MENTORING_QUESTIONS.md
│   ├── QA_LOG.md
│   └── SRC_DESIGN.md
├── outputs/                  # 제출 파일 (output_expN.csv)
├── src/                      # 모듈화 파이프라인
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── infer.py
│   └── run_experiment.py     # 실험 단위 실행 스크립트
└── PROJECT_CONTEXT.md
```

## 시작하기

```bash
pip install -r baseline_code/requirements.txt
cd src && python run_experiment.py
```

## 데이터

`data/train.csv`, `data/test.csv`는 용량 문제로 저장소에 포함하지 않는다.
