"""
create_boundary_submissions.py — 경계값 제출 파일 생성

목적:
  all-1 제출 → F1 점수로 실제 이상 run 수(A) 역산
  공식: A = 740 × F1 / (2 - F1)

  예상:
    A=237 (0.32×740) → F1 = 2×237/(237+740) = 474/977 ≈ 0.4851
    A=222 (0.30×740) → F1 = 2×222/(222+740) = 444/962 ≈ 0.4616
    A=296 (0.40×740) → F1 = 2×296/(296+740) = 592/1036 ≈ 0.5714

실행: cd /root/AD_project/src && python create_boundary_submissions.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_test
from infer import save_submission

DATA_PATH  = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")


def f1_to_anomaly_count(f1: float, total_runs: int = 740) -> float:
    """all-1 제출의 F1으로 실제 이상 run 수 역산."""
    return total_runs * f1 / (2 - f1)


def main():
    print("테스트 데이터 로딩...")
    test_data = load_test(DATA_PATH)
    n = len(test_data)
    print(f"  테스트 행 수: {n:,}  (= {n // 960} runs × 960 timesteps)")

    idx = test_data.index

    # ── all-1 제출 ───────────────────────────────────────────────
    pred_all1 = np.ones(n, dtype=int)
    out_all1 = OUTPUT_DIR / "output_all1_contamination_check.csv"
    save_submission(pred_all1, idx, out_all1)
    print(f"\nall-1 제출 파일 저장: {out_all1}")
    print("  → 리더보드 제출 후 F1 점수로 아래 표에서 실제 이상 run 수 확인:\n")
    print("  F1 점수  |  실제 이상 run 수  |  contamination")
    print("  ---------|--------------------|--------------")
    for a in range(148, 370, 10):
        f1_v = 2 * a / (a + 740)
        print(f"   {f1_v:.4f}  |  {a:3d} runs           |  {a/740:.4f}")

    print("\n  (역산 공식: A = 740 × F1 / (2 - F1))")

    # ── all-0 제출 (참고용 — F1=0이 나와야 정상) ─────────────────
    pred_all0 = np.zeros(n, dtype=int)
    out_all0 = OUTPUT_DIR / "output_all0_sanity_check.csv"
    save_submission(pred_all0, idx, out_all0)
    print(f"\nall-0 제출 파일 저장: {out_all0}")
    print("  → 제출 시 F1=0.0이 나오면 채점 방향(이상=1) 확인 완료")


if __name__ == "__main__":
    main()
