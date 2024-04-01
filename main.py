import dotenv
from omegaconf import DictConfig
import hydra

from src.pipelines.ml_pipeline import train, test, tune

dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="lgbm.yaml")
def main(
    config: DictConfig,
) -> None:
    if config.mode == "train":
        return train(config)
    elif config.mode == "test":
        return test(config)
    elif config.mode == "tune":
        return tune(config)
    else:
        raise ValueError(f"Invalid execution mode: {config.mode}")


if __name__ == "__main__":
    main()
