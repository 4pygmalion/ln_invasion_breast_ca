import sklearn
import argparse
import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyClassifier

from utils import open_yaml
def get_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--input1", help="train_clincal data path", required=True)
    parser.add_argument("-i2", "--input2", help="test_clincal data path", required=True)
    parser.add_argument("-o", "--output", help="clincal data path", required=True)

    return parser.prase()


if __name__ == "__main__":
    ARGS = get_args()
    CONFIG = open_yaml("config.yaml")
    FEATURES = CONFIG["FEATURE"]
    TRAIN_DF = pd.read_csv(ARGS.input1)
    TEST_DF = pd.read_csv(ARGS.input2)

    x, y = TRAIN_DF[FEATURES], TRAIN_DF["N_category"]

    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x, y)
    
    test_clincal_features = imp.transform(TEST_DF[FEATURES])
    np.save(ARGS.output, test_clincal_features)
    print(f"test_clincal_features({ARGS.output}) saved")