from src import config
from src.dataset import load_housing_data
from src.modeling.train import train_model, prepare_data
from src.modeling.predict import (
    save_model,
    load_model,
    calculate_rmse,
    confidence_interval_t_score,
    confidence_interval_bootstrap,
)

seed = config.RANDOM_SEED = 42
test_size = config.TEST_SIZE = 0.2


housing = load_housing_data(config.PATH_DATA_RAW / "housing.csv")

quirk_values = [500001, 500000, 450000, 350000, 280000]

X_train, y_train, X_test, y_test = prepare_data(housing, quirk_values)

model = train_model(X_train, y_train)
save_model(model, config.project_root / "models" / "RF_model.pkl")

model_loaded = load_model(config.project_root / "models" / "RF_model.pkl")

y_test_predictions = model_loaded.predict(X_test)

# Calculate RMSE
rmse = calculate_rmse(y_test, y_test_predictions)
print(f"RMSE: {rmse}")

# Calculate confidence interval using t-score
t_score_interval = confidence_interval_t_score(y_test, y_test_predictions)
print(f"95% confidence interval for RMSE using t-distribution: {t_score_interval}")

# Calculate confidence interval using bootstrap
bootstrap_interval = confidence_interval_bootstrap(y_test, y_test_predictions)
print(f"(Bootstrapped) 95% confidence interval for RMSE: {bootstrap_interval}")
