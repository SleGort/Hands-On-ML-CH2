# Code to create features for modeling
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(
        self, add_bedrooms_per_room=True, columns=None
    ):  # Adding columns as an argument
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is not None:
            self.rooms_ix = self.columns.index("total_rooms")
            self.bedrooms_ix = self.columns.index("total_bedrooms")
            self.population_ix = self.columns.index("population")
            self.households_ix = self.columns.index("households")
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class ImportantAttributesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, count):
        self.count = count
        self.feature_importances_ = None
        self.selected_indices_ = None

    def fit(self, X, y=None):
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(X, y)
        self.feature_importances_ = tree_reg.feature_importances_
        # * The np.argsort reutrns a list of indicies in ascending order ( so the highest value is in the last place)
        self.selected_indices_ = np.argsort(self.feature_importances_)[-self.count :]
        return self

    def transform(self, X, y=None):
        return X[:, self.selected_indices_]
