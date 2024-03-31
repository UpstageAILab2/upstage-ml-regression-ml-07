import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from xgboost import plot_importance

import wandb
from wandb.xgboost import WandbCallback

import matplotlib.pyplot as plt


class XGBArchitecture:
    def __init__(
        self,
        run_name: str,
        model_save_path: str,
        result_summary_path: str,
    ) -> None:
        self.run_name = run_name
        self.model_save_path = model_save_path
        self.result_summary_path = result_summary_path

    def train(
        self,
        data: pd.DataFrame,
        label: pd.Series,
        num_folds: int,
        seed: int,
        is_tuned: bool,
        hparams_save_path: str,
        plt_save_path: str,
    ) -> None:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        if is_tuned == "tuned":
            params = json.load(
                open(f"{hparams_save_path}/best_params.json", "rt", encoding="UTF-8")
            )
            params["verbose"] = -1
        elif is_tuned == "untuned":
            params = {
                "booster": "gbtree",
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            }
        else:
            raise ValueError(f"Invalid is_tuned argument: {is_tuned}")

        wandb.init(
            project="UpStageHousePrice", entity="DimensionSTP", name=self.run_name
        )

        model = xgb.XGBRegressor(**params, random_state=seed)

        rmses = []
        for i, idx in enumerate(tqdm(kf.split(data, label))):
            train_data, train_label = data.loc[idx[0]], label.loc[idx[0]]
            val_data, val_label = data.loc[idx[1]], label.loc[idx[1]]

            model.fit(
                train_data, train_label, callbacks=[WandbCallback(log_model=True)]
            )

            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            model.save_model(f"{self.model_save_path}/fold{i}.model")

            pred = model.predict(val_data)
            rmse = np.sqrt(mean_squared_error(val_label, pred))
            rmses.append(rmse)
        avg_rmse = np.mean(rmses)
        print(f"average RMSE : {avg_rmse}")

        result = {
            "Model type": "XGBoost",
            "Used features": data.columns.tolist(),
            "Num Kfolds": num_folds,
            "RMSE": avg_rmse,
        }
        result_df = pd.DataFrame.from_dict(result, orient="index").T

        if not os.path.exists(self.result_summary_path):
            os.makedirs(self.result_summary_path)

        result_file = f"{self.result_summary_path}/result_summary.csv"
        if os.path.isfile(result_file):
            original_result_df = pd.read_csv(result_file)
            new_result_df = pd.concat(
                [original_result_df, result_df], ignore_index=True
            )
            new_result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )
        else:
            result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )

        fig, ax = plt.subplots(figsize=(10, 12))
        plot_importance(model, ax=ax)
        if not os.path.exists(plt_save_path):
            os.makedirs(plt_save_path)
        plt.savefig(f"{plt_save_path}/num_folds{num_folds}-rmse{avg_rmse}.png")

    def test(
        self,
        data: pd.DataFrame,
        submission_save_path: str,
        submission_save_name: str,
    ) -> None:
        pred_mean = np.zeros((len(data),))
        for model_file in tqdm(os.listdir(self.model_save_path)):
            model = xgb.XGBRegressor()
            model.load_model(f"{self.model_save_path}/{model_file}")
            pred = model.predict(data) / len((os.listdir(self.model_save_path)))
            pred_mean += pred
        submission = pd.DataFrame(pred_mean.astype(int), columns=["target"])
        if not os.path.exists(submission_save_path):
            os.makedirs(submission_save_path)
        submission.to_csv(
            f"{submission_save_path}/{submission_save_name}.csv", index=False
        )
