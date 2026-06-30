"""설계 근거: docs/SRC_DESIGN.md 참고.

이 스크립트 실행 = 실험 하나를 재현하는 것. 실행 시점의 git commit이
docs/EXPERIMENT_LOG.md의 해당 항목(커밋 해시 포함)과 1:1로 대응해야 한다.

새 실험을 시작할 때는 EXP_NAME과 바꾸려는 파라미터(예: CONTAMINATION)를 같이 바꿀 것
— 파일명과 로그 번호가 어긋나지 않게 하기 위함.

실행: cd src && python run_experiment.py
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_test, load_train
from infer import predict_labels, save_submission
from model import train_isolation_forest
from preprocess import select_features

RANDOM_SEED = 42
DATA_PATH = Path("/root/AD_project/data")
OUTPUT_DIR = Path("/root/AD_project/outputs")

EXP_NAME = "exp2"
CONTAMINATION = 0.23

# 참고: Exp 0 역산 결과, 모델이 예측한 positive 비율(14.8%)이 추정 실제 비율(32.2%)보다
# 훨씬 낮았다 (docs/EXPERIMENT_LOG.md 참고). 이 값과 비교해 보며 다음 파라미터를 정한다.
ESTIMATED_TRUE_POSITIVE_RATE = 0.322


def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    train_data = load_train(DATA_PATH)
    train_X = select_features(train_data)
    model = train_isolation_forest(train_X, random_state=RANDOM_SEED, contamination=CONTAMINATION)

    test_data = load_test(DATA_PATH)
    test_X = select_features(test_data)
    pred = predict_labels(model, test_X)

    output_path = OUTPUT_DIR / f"output_{EXP_NAME}.csv"
    save_submission(pred, test_X.index, output_path)

    counts = pd.Series(pred).value_counts().sort_index()
    pred_positive_rate = counts.get(1, 0) / len(pred)
    print(f"[{EXP_NAME}] saved -> {output_path}")
    print(f"label counts:\n{counts}")
    print(
        f"predicted positive rate: {pred_positive_rate:.4f} "
        f"(estimated true rate: {ESTIMATED_TRUE_POSITIVE_RATE:.4f})"
    )


if __name__ == "__main__":
    main()
