# Data Project Template

## Cookiecutter Data Science
This project template is a simplified version of the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org) template, created to suit the needs of Datalumina and made available as a GitHub template.

## Duplicating the .env File
To set up your environment variables, you need to duplicate the `.env.example` file and rename it to `.env`. You can do this manually or using the following terminal command:

```bash
cp .env.example .env # Linux, macOS, Git Bash, WSL
copy .env.example .env # Windows Command Prompt
```

This command creates a copy of `.env.example` and names it `.env`, allowing you to configure your environment variables specific to your setup.

## Project Organization
```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third-party sources
│   ├── interim        <- Intermediate data that has been transformed, data in this directory
│   │                     is what would be called for any feature engineering. (separate files)
│   ├── processed      <- The final, canonical data sets for modeling
│   ├── raw            <- The original, immutable data dump
│   └── test_set       <- Final test set, kept locked until model selection is complete.
│                         Only to be used for final generalization performance testing.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                         <- Source code for this project
   ├── __init__.py             <- Makes src a Python module
   ├── config.py               <- Store useful variables and configuration
   ├── dataset.py              <- Scripts to download or generate data
   ├── features.py             <- Code to create features for modeling
   ├── modeling                
   │   ├── __init__.py 
   │   ├── predict.py          <- Code to run model inference with trained models          
   │   └── train.py            <- Code to train models
   ├── plots.py                <- Code to create visualizations 
   └── services                <- Service classes to connect with external platforms, tools, or APIs
       └── __init__.py

                         
```
## Workflow

Naming convention is a number (for ordering), the creator's initials, and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`

Processed folder = Train data -> Split into train and validation set -> Validation set is to be used for evaluating generalizability of the models (model selection step) -> Choose the best model -> Train that model on (train set + validation set) -> Use the trained model on test_set data for the final test of the generalization performance -> Move to production?

--------
