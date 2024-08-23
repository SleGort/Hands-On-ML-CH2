# Scripts to download or generate data
import os
import tarfile
import urllib.request
import pandas as pd


def check_and_load_data(data_path):
    """
    Checks if the data exists at the specified location. If not, calls the main() function to load the data.

    Args:
    - data_path: The path where the data should be located.

    Returns:
    - Boolean: True if data exists after the function, False otherwise.
    """
    if os.path.exists(data_path):
        print(f"Data found at {data_path}.")
        return True
    else:
        print(f"Data not found at {data_path}. Loading data...")
        main()

        # Check again if the data was successfully loaded
        if os.path.exists(data_path):
            print(f"Data successfully loaded and found at {data_path}.")
            return True
        else:
            print(f"Failed to load data to {data_path}.")
            return False


def fetch_housing_data(housing_url, housing_path):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    return pd.read_csv(housing_path)


def main():
    import config

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = config.PATH_DATA_RAW
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)


if __name__ == "__main__":
    main()
