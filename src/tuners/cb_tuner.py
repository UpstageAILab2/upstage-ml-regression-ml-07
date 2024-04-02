import os
from typing import Tuple
import json
import warnings

from omegaconf import DictConfig

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import catboost as cb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from typing import List


class CBTuner:
    def __init__(
        self,
        hparams: DictConfig,
        data: pd.DataFrame,
        label: pd.Series,
        split_size: float,
        num_folds: int,
        num_trials: int,
        seed: int,
        tuning_way: str,
        hparams_save_path: str,
    ) -> None:
        self.hparams = hparams
        self.data = data
        self.label = label
        self.split_size = split_size
        self.num_folds = num_folds
        self.num_trials = num_trials
        self.seed = seed
        self.tuning_way = tuning_way
        self.hparams_save_path = hparams_save_path

    def __call__(self) -> None:
        if self.tuning_way == "original":
            study = optuna.create_study(
                direction="minimize",
                sampler=TPESampler(seed=self.seed),
                pruner=HyperbandPruner(),
            )
            study.optimize(self.optuna_objective, n_trials=self.num_trials)
            trial = study.best_trial
            best_score = trial.value
            best_params = trial.params
        else:
            raise ValueError("Invalid tuning way")

        print(f"Best score : {best_score}")
        print(f"Parameters : {best_params}")

        if not os.path.exists(self.hparams_save_path):
            os.makedirs(self.hparams_save_path, exist_ok=True)

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(best_params, json_file)

    @property
    def cat_features(self) -> List[str]:
        return [
            column
            for column in self.data.columns
            if self.data[column].dtype == "object"
        ]

    def get_split_dataset(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        train_data, val_data, train_label, val_label = train_test_split(
            self.data,
            self.label,
            test_size=self.split_size,
            random_state=self.seed,
            shuffle=True,
        )
        return (train_data, val_data, train_label, val_label)

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        params = {
            "iterations": trial.suggest_int(
                "iterations",
                self.hparams.iterations.low,
                self.hparams.iterations.high,
                step=self.hparams.iterations.step,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.hparams.learning_rate.low,
                self.hparams.learning_rate.high,
                log=self.hparams.learning_rate.log,
            ),
            "depth": trial.suggest_int(
                "depth", self.hparams.depth.low, self.hparams.depth.high
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                self.hparams.l2_leaf_reg.low,
                self.hparams.l2_leaf_reg.high,
                log=self.hparams.l2_leaf_reg.log,
            ),
            "model_size_reg": trial.suggest_float(
                "model_size_reg",
                self.hparams.model_size_reg.low,
                self.hparams.model_size_reg.high,
                log=self.hparams.model_size_reg.log,
            ),
            "rsm": trial.suggest_float(
                "rsm", self.hparams.rsm.low, self.hparams.rsm.high
            ),
            "subsample": trial.suggest_float(
                "subsample", self.hparams.subsample.low, self.hparams.subsample.high
            ),
            "border_count": trial.suggest_int(
                "border_count",
                self.hparams.border_count.low,
                self.hparams.border_count.high,
            ),
            "feature_border_type": trial.suggest_categorical(
                "feature_border_type", self.hparams.feature_border_type
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", self.hparams.bootstrap_type
            ),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", self.hparams.grow_policy
            ),
            "leaf_estimation_method": trial.suggest_categorical(
                "leaf_estimation_method", self.hparams.leaf_estimation_method
            ),
            "random_strength": trial.suggest_int(
                "random_strength",
                self.hparams.random_strength.low,
                self.hparams.random_strength.high,
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature",
                self.hparams.bagging_temperature.low,
                self.hparams.bagging_temperature.high,
            ),
        }

        if params["bootstrap_type"] == "Bayesian":
            del params["subsample"]
        else:
            del params["bagging_temperature"]

        train_data, val_data, train_label, val_label = self.get_split_dataset()

        model = cb.CatBoostRegressor(**params)

        model.fit(train_data, train_label, cat_features=self.cat_features)
        pred = model.predict(val_data)
        score: float = np.sqrt(mean_squared_error(val_label, pred)).item()

        return score
