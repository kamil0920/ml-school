
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pickle import dump
from pathlib import Path


# This is the location where the SageMaker Processing job
# will save the input dataset.
BASE_DIR = "/opt/ml/processing"
DATA_FILEPATH_TRAIN = Path(BASE_DIR) / "input" / "mnist_train" / "mnist_train.csv"
DATA_FILEPATH_TEST = Path(BASE_DIR) / "input" / "mnist_test" / "mnist_test.csv"


def save_splits(base_dir, train, validation, test):
    """
    One of the goals of this script is to output the three
    dataset splits. This function will save each of these
    splits to disk.
    """

    train_path = Path(base_dir) / "train"
    validation_path = Path(base_dir) / "validation"
    test_path = Path(base_dir) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(validation_path / "validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)


def save_pipeline(base_dir, pipeline):
    """
    Saves the Scikit-Learn pipeline that we used to
    preprocess the data.
    """
    pipeline_path = Path(base_dir) / "pipeline"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / "pipeline.pkl", 'wb'))


def generate_baseline(base_dir, X_train, y_train):
    """
    Generates a baseline for our model using the train set.
    It saves the baseline in a JSON file where every line is
    a JSON object.
    """
    baseline_path = Path(base_dir) / "baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)

    df = X_train.copy()
    df["groundtruth"] = y_train

    df.to_json(baseline_path / "baseline.json", orient='records', lines=True)


def preprocess(base_dir, data_filepath_train, data_filepath_test):
    """
    Preprocesses the supplied raw dataset and splits it into a train, validation,
    and a test set.
    """

    df_train = pd.read_csv(data_filepath_train, nrows=7200)
    df_test = pd.read_csv(data_filepath_test, nrows=2000)

    numerical_columns = df_train.select_dtypes(include=['number']).drop(['label'], axis=1).columns

    numerical_preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_preprocessor, numerical_columns),
        ]
    )

    X_train = df_train.copy()
    y_train = df_train['label']
    columns = list(X_train.drop(['label'], axis=1).columns)

    X_train, X_validation, y_train, y_validation =  train_test_split(X_train, y_train, test_size=0.2, random_state=12)
    X_test = df_test.copy()

    y_train = X_train.label
    y_validation = X_validation.label
    y_test = X_test.label

    X_train.drop(["label"], axis=1, inplace=True)
    X_validation.drop(["label"], axis=1, inplace=True)
    X_test.drop(["label"], axis=1, inplace=True)

    X_train = pd.DataFrame(X_train, columns=columns)
    X_validation = pd.DataFrame(X_validation, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)

    y_train = y_train.astype(int)
    y_validation = y_validation.astype(int)
    y_test = y_test.astype(int)

    # Let's use the train set to generate a baseline that we can
    # later use to measure the quality of our model. This baseline
    # will use the original data.
    generate_baseline(base_dir, X_train, y_train)

    # Transform the data using the Scikit-Learn pipeline.
    X_train = preprocessor.fit_transform(X_train)
    X_validation = preprocessor.transform(X_validation)
    X_test = preprocessor.transform(X_test)

    train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
    validation = np.concatenate((X_validation, np.expand_dims(y_validation, axis=1)), axis=1)
    test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)

    save_splits(base_dir, train, validation, test)
    save_pipeline(base_dir, pipeline=preprocessor)


if __name__ == "__main__":
    preprocess(BASE_DIR, DATA_FILEPATH_TRAIN, DATA_FILEPATH_TEST)
