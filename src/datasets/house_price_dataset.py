from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class HousePriceDataset:
    def __init__(
        self,
        mode: str,
        dataset_name: str,
        df_path: str,
        label_column_name: str,
        scale: str,
    ) -> None:
        self.mode = mode
        self.dataset_name = dataset_name
        self.df_path = df_path
        self.label_column_name = label_column_name
        self.scale = scale

    def __call__(self) -> Tuple[pd.DataFrame, pd.Series]:
        dataset = self.load_dataset()
        dataset = self.preprocess_certain_features(dataset)
        dataset = self.interpolate_dataset(dataset)
        data, label = self.get_preprocessed_dataset(dataset)
        return (data, label)

    def load_dataset(self) -> pd.DataFrame:
        if self.mode == "train" or self.mode == "test":
            dataset = pd.read_csv(f"{self.df_path}/{self.dataset_name}_{self.mode}.csv")
        elif self.mode == "tune":
            dataset = pd.read_csv(f"{self.df_path}/{self.dataset_name}_train.csv")
        else:
            raise ValueError(f"Invalid execution mode: {self.mode}")
        return dataset

    def preprocess_certain_features(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        dataset.columns = [
            column.replace("㎡", "")
            .replace("'", "")
            .replace('"', "")
            .replace("{", "")
            .replace("}", "")
            .replace("[", "")
            .replace("]", "")
            .replace(":", "")
            .replace(",", "")
            for column in dataset.columns
        ]
        return dataset

    def get_columns_by_types(
        self,
        dataset: pd.DataFrame,
    ) -> Tuple[List[str], List[str]]:
        continuous_columns = []
        categorical_columns = []
        for column in dataset.columns:
            if pd.api.types.is_numeric_dtype(dataset[column]):
                continuous_columns.append(column)
            else:
                categorical_columns.append(column)
        return continuous_columns, categorical_columns

    def interpolate_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        continuous_columns, categorical_columns = self.get_columns_by_types(dataset)
        dataset[continuous_columns] = dataset[continuous_columns].interpolate(
            method="linear", axis=0
        )
        dataset[categorical_columns] = dataset[categorical_columns].fillna("NULL")
        return dataset

    def encode_categorical_features(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        _, categorical_columns = self.get_columns_by_types(dataset)
        for column in categorical_columns:
            label_encoder = LabelEncoder()
            dataset[column] = label_encoder.fit_transform(dataset[column].astype(str))
        return dataset

    def get_preprocessed_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if self.mode == "train" or self.mode == "tune":
            data = dataset.drop([self.label_column_name], axis=1)
            label = dataset[self.label_column_name]
        elif self.mode == "test":
            data = dataset
            label = 0
        else:
            raise ValueError(f"Invalid execution mode: {self.mode}")

        if self.scale == "unscale":
            pass
        elif self.scale == "standard":
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            data = pd.DataFrame(scaled_data, columns=data.columns)
        elif self.scale == "min-max":
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            data = pd.DataFrame(scaled_data, columns=data.columns)
        else:
            raise ValueError(f"Invalid execution scale: {self.scale}")
        return (data, label)
