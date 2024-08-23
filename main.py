from datetime import datetime

from src import config
from src.dataset import check_and_load_data, load_housing_data
from src.modeling.predict import (
    calculate_rmse,
    confidence_interval_bootstrap,
    confidence_interval_t_score,
    load_model,
    save_model,
)
from src.modeling.train import prepare_data, train_model

# Get the current local time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

data_path = config.project_root / "data" / "raw" / "housing.csv"
if check_and_load_data(data_path):
    print("Data is ready for processing.")
else:
    print("Data loading failed. Please check the dataset.py script.")

housing = load_housing_data(config.PATH_DATA_RAW / "housing.csv")

quirk_values = [500001, 500000, 450000, 350000, 280000]

X_train, y_train, X_test, y_test = prepare_data(housing, quirk_values)

print(f"[{current_time}] Started training model")

model = train_model(X_train, y_train)

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{current_time}] Model training finished")

save_model(model, config.project_root / "models" / "RF_model.pkl")

model_loaded = load_model(config.project_root / "models" / "RF_model.pkl")

y_test_predictions = model_loaded.predict(X_test)

# Calculate RMSE
rmse = calculate_rmse(y_test, y_test_predictions)
print(f"RMSE: {rmse}")

# Calculate confidence interval using t-distribution
t_score_interval = confidence_interval_t_score(y_test, y_test_predictions)
print(f"95% confidence interval for RMSE using t-distribution: {t_score_interval}")

# Calculate confidence interval using bootstrap
bootstrap_interval = confidence_interval_bootstrap(y_test, y_test_predictions)
print(f"(Bootstrapped) 95% confidence interval for RMSE: {bootstrap_interval}")
