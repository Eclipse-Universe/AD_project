"""설계 근거: docs/SRC_DESIGN.md 참고. baseline.ipynb cell 10, 55와 동일한 역할(csv 로드만)."""
from pathlib import Path

import pandas as pd


def load_train(path: Path) -> pd.DataFrame:
    # TODO: train.csv를 읽어 DataFrame으로 반환 (baseline.ipynb cell 10)
    raise NotImplementedError


def load_test(path: Path) -> pd.DataFrame:
    # TODO: test.csv를 읽어 DataFrame으로 반환 (baseline.ipynb cell 55)
    raise NotImplementedError
