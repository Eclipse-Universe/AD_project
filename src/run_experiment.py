"""설계 근거: docs/SRC_DESIGN.md 참고.

이 스크립트 실행 = 실험 하나를 재현하는 것. 실행 시점의 git commit이
docs/EXPERIMENT_LOG.md의 해당 항목과 1:1로 대응해야 한다.

TODO:
- data_loader -> preprocess -> model -> infer 순서로 연결
- contamination 등 실험별 설정값을 이 파일 상단 상수 또는 CLI 인자로 노출
- import 방식(상대 경로 vs 패키지 설치)은 직접 결정할 것 — 정답은 없음
"""
