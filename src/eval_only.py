import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

from src.utils import log
from src.utils.enviroment import get_config

# Monkey-patch torch.load (PyTorch 2.6+ compatibility)
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def load_model_config(model_name: str, project_config: dict) -> dict:
    config_base = Path("src/configs")
    for category in ["neural", "baselines", "factorization"]:
        config_file = config_base / category / f"{model_name.lower()}.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                model_config = yaml.safe_load(f)
            break
    else:
        raise FileNotFoundError(f"Config not found for model: {model_name}")

    config_dict = {**project_config, **model_config}
    config_dict["data_path"] = project_config["data_path"]
    return config_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved RecBole checkpoint")
    parser.add_argument("--model", required=True, help="Model name (e.g., TransRec)")
    parser.add_argument("--checkpoint", required=True, help="Path to saved .pth checkpoint")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (default: from config/project_config.yaml)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override eval_batch_size for evaluation",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (e.g., cpu or cuda)",
    )
    parser.add_argument(
        "--wandb-group",
        default="run_final",
        help="W&B run group (default: run_final)",
    )
    args = parser.parse_args()

    project_config = get_config()
    config_dict = load_model_config(args.model, project_config)

    dataset_name = args.dataset or project_config["dataset"]
    config_dict["dataset"] = dataset_name
    config_dict["show_progress"] = True

    if args.eval_batch_size is not None:
        config_dict["eval_batch_size"] = args.eval_batch_size

    if args.device is not None:
        config_dict["device"] = args.device

    if config_dict.get("log_wandb", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["WANDB_NAME"] = f"Eval_{args.model}_{timestamp}"
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group

    log(f"Evaluating checkpoint: {args.checkpoint}")
    log(f"Model: {args.model} | Dataset: {dataset_name}")
    if args.eval_batch_size is not None:
        log(f"Eval batch size override: {args.eval_batch_size}")
    if args.device is not None:
        log(f"Device override: {args.device}")

    config = Config(model=args.model, config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    from recbole.utils import get_model
    model = get_model(args.model)(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    result = trainer.evaluate(
        test_data, model_file=str(args.checkpoint), show_progress=True
    )

    if config["log_wandb"]:
        import wandb
        if wandb.run is not None:
            wandb.log({f"test_{k}": v for k, v in result.items()})

    log(f"Resultados (test): {result}")


if __name__ == "__main__":
    main()
