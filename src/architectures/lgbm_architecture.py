import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from lightgbm import plot_importance

import wandb
from wandb.lightgbm import wandb_callback, log_summary

import matplotlib.pyplot as plt


class LGBMArchitecture:
    def __init__(
        self,
        run_name: str,
        model_save_path: str,
        result_summary_path: str,
        wandb_project: str,
        wandb_entity: str,
    ) -> None:
        self.run_name = run_name
        self.model_save_path = model_save_path
        self.result_summary_path = result_summary_path

        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

    def train(
        self,
        data: pd.DataFrame,
        label: pd.Series,
        num_folds: int,
        seed: int,
        is_tuned: str,
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
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": "rmse",
                "seed": seed,
            }
        else:
            raise ValueError(f"Invalid is_tuned argument: {is_tuned}")

        wandb.init(
            project=self.wandb_project, entity=self.wandb_entity, name=self.run_name
        )

        rmses = []
        for i, idx in enumerate(tqdm(kf.split(data, label))):
            train_data, train_label = data.loc[idx[0]], label.loc[idx[0]]
            val_data, val_label = data.loc[idx[1]], label.loc[idx[1]]
            train_dataset = lgb.Dataset(train_data, train_label)
            val_dataset = lgb.Dataset(val_data, val_label)

            model = lgb.train(
                params,
                train_dataset,
                valid_sets=[train_dataset, val_dataset],
                valid_names=("validation"),
                callbacks=[wandb_callback()],
            )
            log_summary(model, save_model_checkpoint=True)

            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            model.save_model(f"{self.model_save_path}/fold{i}.txt")

            pred = model.predict(val_data)
            rmse = np.sqrt(mean_squared_error(val_label, pred))
            rmses.append(rmse)
        avg_rmse = np.mean(rmses)
        print(f"average RMSE : {avg_rmse}")

        result = {
            "Model type": "LightGBM",
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
            model = lgb.Booster(model_file=f"{self.model_save_path}/{model_file}")
            pred = model.predict(data) / len((os.listdir(self.model_save_path)))
            pred_mean += pred
        submission = pd.DataFrame(pred_mean.astype(int), columns=["target"])
        if not os.path.exists(submission_save_path):
            os.makedirs(submission_save_path)
        submission.to_csv(
            f"{submission_save_path}/{submission_save_name}.csv", index=False
        )
