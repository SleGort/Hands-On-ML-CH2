from src.dataset import load_housing_data
from src import config

housing = load_housing_data(config.PATH_DATA_RAW / "housing.csv")
housing.head()
