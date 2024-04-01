import os
from re import I
from typing import Dict, Any, Tuple
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error

import catboost as cb

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


class CBTuner:
    def __init__(
        self,
        hparams: Dict[str, Any],
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
        params = dict()
        params["random_seed"] = self.seed
        if self.hparams.objective:
            params["objective"] = self.hparams.objective
        if self.hparams.loss_function:
            params["loss_function"] = self.hparams.loss_function
        if self.hparams.boosting_type:
            params["boosting_type"] = trial.suggest_categorical(
                name="boosting_type",
                choices=self.hparams.boosting_type,
            )
        if self.hparams.learning_rate:
            params["learning_rate"] = trial.suggest_float(
                name="learning_rate",
                low=self.hparams.learning_rate.low,
                high=self.hparams.learning_rate.high,
                log=self.hparams.learning_rate.log,
            )
        if self.hparams.n_estimators:
            params["n_estimators"] = trial.suggest_int(
                name="n_estimators",
                low=self.hparams.n_estimators.low,
                high=self.hparams.n_estimators.high,
                log=self.hparams.n_estimators.log,
            )
        if self.hparams.min_child_samples:
            params["min_child_samples"] = trial.suggest_int(
                name="min_child_samples",
                low=self.hparams.min_child_samples.low,
                high=self.hparams.min_child_samples.high,
                log=self.hparams.min_child_samples.log,
            )
        if self.hparams.subsample:
            params["subsample"] = trial.suggest_uniform(
                name="subsample",
                low=self.hparams.subsample.low,
                high=self.hparams.subsample.high,
            )
        train_data, val_data, train_label, val_label = self.get_split_dataset()

        model = cb.CatBoostRegressor(**params)

        model.fit(train_data, train_label)

        pred = model.predict(val_data)
        score = np.sqrt(mean_squared_error(val_label, pred))
        return score
