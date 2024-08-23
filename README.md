# Machine Learning Model Training and Inference

This project is designed to facilitate the training, evaluation, and deployment of machine learning models. The project is structured to allow easy data preprocessing, model training, and model inference. 

## Project Structure

```
├── data/      
│   └── raw/               # The original, immutable data dump
│       ├── housing.csv    # Example raw data file
│
├── models/                # Trained models (the model itself is not included)
│
├── notebooks/             # Jupyter notebooks for exploration and analysis
│   ├── experimentation.ipynb # Exploratory code to identify patterns
│   └── final.ipynb           # Less messy code for structured program flow
├── reports/               
│   └── figures/           # Generated graphics and figures 
│
├── src/                   # Source code for use in this project
│   ├── dataset.py         # Script for loading data    
│   ├── features.py        # Feature engineering classess   
│   ├── modeling/
│   │   ├── train.py       # Script for pre processing and training models
│   │   └── predict.py     # Script for assessing predictions with trained models
│   ├── config.py          # Configuration settings
│   └── plots.py           # Code for plotting
│             
│
├── main.py                # Main script to run the project
├── .gitignore             # Git ignore file
├── README.md              # Project README
└── requirements.txt       # Python dependencies
```

## Getting Started

### Prerequisites

Before running this project, make sure you have Python installed. You can install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### Setting Up the Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SleGort/Hands-On-ML-CH2
   cd your-repo-name
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data:**
   - Place your raw data files in the `data/raw/` directory.
   - If the data is not present, the program will attempt to download it automatically when you run the scripts.

## Usage

### 1. Data Preparation

The `dataset.py` script handles data loading and preprocessing. To check if the data is present and load it if necessary, use:

```python
from src.data.dataset import check_and_load_data

data_path = "data/raw/housing.csv"
if check_and_load_data(data_path):
    print("Data is ready for processing.")
```

### 2. Model Training

To train a machine learning model, run the `train.py` script in the `modeling` directory. This script handles the entire model training process, including hyperparameter tuning using `GridSearchCV`.

```python
from src.modeling.train import prepare_data, train_model

# Assuming `housing` DataFrame is already loaded
X_train, y_train, X_test, y_test = prepare_data(housing)

# Train the model
best_model = train_model(X_train, y_train)
```

### 3. Saving and Loading Models

The trained models can be saved and loaded using functions defined in the `predict.py` script.

```python
from src.modeling.predict import save_model, load_model

save_model(best_model, "models/my_model.pkl")
loaded_model = load_model("models/my_model.pkl")
```
## Acknowledgments

- This project was inspired from the 2nd Chapter of the book Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow. (Aurelien Geron, 2nd edition)
