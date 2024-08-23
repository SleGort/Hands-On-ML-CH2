import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# sys.path.append(os.path.dirname(os.getcwd()))

from ..features import CombinedAttributesAdder, ImportantAttributesSelector

# Define column names for numerical and categorical data
col_names = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

# Pipeline for numerical attributes
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder(columns=col_names)),
        ("std_scaler", StandardScaler()),
    ]
)

# Categorical attributes and known categories
cat_attribs = ["ocean_proximity"]
known_categories = [["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]]

# Full preprocessing pipeline
full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, col_names),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", categories=known_categories),
            cat_attribs,
        ),
    ]
)

# Final pipeline combining preprocessing, feature selection, and model training
final_pipeline = Pipeline(
    [
        ("preparation", full_pipeline),
        ("feature_selection", ImportantAttributesSelector(count=6)),
        (
            "regressor",
            RandomForestRegressor(random_state=42),
        ),
    ]
)


def create_income_category(housing):
    """
    Create an income category attribute for stratified sampling.

    Args:
    - housing: The entire dataset as a DataFrame.

    Returns:
    - housing: DataFrame with an additional 'income_cat' column.
    """
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def split_data(housing, test_size=0.2, seed=42):
    """
    Split the data into training and test sets using stratified sampling based on income category.

    Args:
    - housing: The entire dataset as a DataFrame.
    - test_size: The proportion of the dataset to include in the test split.
    - seed: The random seed for reproducibility.

    Returns:
    - strat_train_set: The training set after stratified splitting.
    - strat_test_set: The test set after stratified splitting.
    """
    # Ensure the income category exists
    if "income_cat" not in housing.columns:
        housing = create_income_category(housing)

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Drop the 'income_cat' column as it's no longer needed
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def filter_quirk_values(housing_stratified, quirk_values=None):
    """
    Filter out rows from the DataFrame that contain specific quirk values in 'median_house_value'.

    Args:
    - housing_stratified: The stratified training set as a DataFrame.
    - quirk_values: A list of quirk values to be removed from the dataset.

    Returns:
    - housing_filtered: The filtered DataFrame with quirk values removed.
    """
    if quirk_values is None:
        quirk_values = [500001, 500000, 450000, 350000, 280000]

    housing_filtered = housing_stratified[
        ~housing_stratified["median_house_value"].isin(quirk_values)
    ]

    return housing_filtered


def train_model(X_train, y_train, pipeline=final_pipeline):
    """
    Train the model using GridSearchCV to find the best hyperparameters.

    Args:
    - X_train: Training feature set.
    - y_train: Training target set.
    - pipeline: The machine learning pipeline including preprocessing, feature selection, and model.

    Returns:
    - The best model found by GridSearchCV.
    """
    param_grid = {
        # Number of features to select
        "feature_selection__count": [4, 6, 8],
        "preparation__num__attribs_adder__add_bedrooms_per_room": [True, False],
        # RandomForestRegressor parameters
        "regressor__n_estimators": [10, 50, 100],
        "regressor__max_features": [4, 6, 8],
    }

    # Perform grid search with cross-validation
    grid_pipeline = GridSearchCV(
        pipeline, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_pipeline.fit(X_train, y_train)

    # Output best parameters and return best model
    print("Best parameters found by GridSearchCV:")
    print(grid_pipeline.best_params_)

    best_model = grid_pipeline.best_estimator_
    return best_model


def prepare_data(housing, quirk_values=None, test_size=0.2, seed=42):
    """
    Full data preparation process including stratification, splitting, filtering, and returning ready-to-use datasets.

    Args:
    - housing: The entire dataset as a DataFrame.
    - quirk_values: A list of quirk values to be removed from the dataset.
    - test_size: The proportion of the dataset to include in the test split.
    - seed: The random seed for reproducibility.

    Returns:
    - X_train: Training feature set after filtering and preparation.
    - y_train: Training target set.
    - X_test: Test feature set after preparation.
    - y_test: Test target set.
    """
    # Create income categories for stratified sampling
    housing = create_income_category(housing)

    # Split data into training and test sets
    strat_train_set, strat_test_set = split_data(housing, test_size, seed)
    housing_stratified = strat_train_set.copy()

    # Filter out quirk values
    housing_filtered = filter_quirk_values(housing_stratified, quirk_values)

    # Split the data into features and target
    X_train = housing_filtered.drop("median_house_value", axis=1)
    y_train = housing_filtered["median_house_value"].copy()

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    return X_train, y_train, X_test, y_test
