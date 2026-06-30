"""설계 근거: docs/SRC_DESIGN.md 참고.

이 스크립트 실행 = 실험 하나를 재현하는 것. 실행 시점의 git commit이
docs/EXPERIMENT_LOG.md의 해당 항목과 1:1로 대응해야 한다.

실행: cd src && python run_experiment.py
"""
import random
from pathlib import Path

import numpy as np

from data_loader import load_test, load_train
from infer import predict_labels, save_submission
from model import train_isolation_forest
from preprocess import select_features

RANDOM_SEED = 42
DATA_PATH = Path("/root/AD_project/data")
OUTPUT_PATH = Path(__file__).parent / "output.csv"

# Exp 0(baseline) 재현 — 'auto'는 baseline의 기본값과 동일.
# Exp 1 후보: docs/EXPERIMENT_LOG.md에서 합의한 대로, 이 값을 실제 추정 비율(~0.32)로 바꿔본다.
CONTAMINATION = "auto"


def main():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    train_data = load_train(DATA_PATH)
    train_X = select_features(train_data)
    model = train_isolation_forest(train_X, random_state=RANDOM_SEED, contamination=CONTAMINATION)

    test_data = load_test(DATA_PATH)
    test_X = select_features(test_data)
    pred = predict_labels(model, test_X)
    save_submission(pred, test_X.index, OUTPUT_PATH)


if __name__ == "__main__":
    main()
