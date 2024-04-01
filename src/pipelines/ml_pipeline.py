from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..utils.ml_setup import MLSetUp

from ..tuners.cb_tuner import CBTuner
from ..tuners.lgbm_tuner import LGBMTuner
from ..tuners.xgb_tuner import XGBTuner


def train(
    config: DictConfig,
) -> None:
    ml_setup = MLSetUp(config)

    dataset = ml_setup.get_dataset()
    architecture = ml_setup.get_architecture()

    data, label = dataset()
    architecture.train(
        data=data,
        label=label,
        num_folds=config.num_folds,
        seed=config.seed,
        is_tuned=config.is_tuned,
        hparams_save_path=config.hparams_save_path,
        plt_save_path=config.plt_save_path,
    )


def test(
    config: DictConfig,
) -> None:
    ml_setup = MLSetUp(config)

    dataset = ml_setup.get_dataset()
    architecture = ml_setup.get_architecture()

    data, _ = dataset()
    architecture.test(
        data=data,
        submission_save_path=config.submission_save_path,
        submission_save_name=config.submission_save_name,
    )


def tune(
    config: DictConfig,
) -> None:
    ml_setup = MLSetUp(config)

    dataset = ml_setup.get_dataset()

    data, label = dataset()
    tuner: Union[LGBMTuner, XGBTuner, CBTuner] = instantiate(
        config.tuner, data=data, label=label
    )
    tuner()
