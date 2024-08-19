import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from src import config
from src.dataset import load_housing_data

seed = config.RANDOM_SEED = 42
test_size = config.TEST_SIZE = 0.2

housing = load_housing_data(config.PATH_DATA_RAW / "housing.csv")

# * Using income categories to create stratified random sample for the test set
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
