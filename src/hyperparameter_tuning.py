"""Hyperparameter tuning helper built on RecBole HyperTuning.

This script wraps the workflow documented in
https://recbole.io/docs/user_guide/usage/parameter_tuning.html and adapts it to
Fermi's configuration structure. It automatically:

1. Carrega as configs base do projeto + modelo
2. Lê o search space em `src/configs/tuning/<modelo>_space.yaml`
3. Roda o HyperTuning (HyperOpt) e exporta os melhores parâmetros
"""
from __future__ import annotations

import argparse
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from hyperopt import hp
from recbole.quick_start import objective_function
from recbole.trainer import HyperTuning

from src.utils.enviroment import get_config

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

MODEL_CONFIG_DIRS = ["neural", "factorization", "baselines"]
DEFAULT_SPACE = {
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
}


class RecBoleHyperparameterTuner:
    """Utility class to orchestrate HyperTuning runs."""

    def __init__(
        self,
        model_name: str,
        dataset: str | None = None,
        algo: str = "random",
        max_evals: int = 20,
        early_stop: int = 10,
        output_dir: str | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        self.dataset = dataset or get_config("dataset")
        self.algo = algo
        self.max_evals = max_evals
        self.early_stop = early_stop
        self.project_config = get_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_results_dir = Path(
            self.project_config.get("output", {}).get("results_dir", "outputs/results")
        )
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = base_results_dir / "tuning" / self.timestamp
        self.output_dir = self.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_config_path = self._resolve_model_config()
        self.base_config = self._build_base_config()
        self.fixed_config_path: Optional[Path] = None
        self.search_space = self._load_search_space()

        self.logger.info("Model: %s", self.model_name)
        self.logger.info("Dataset: %s", self.dataset)
        self.logger.info("Output dir: %s", self.output_dir)
        self.logger.info("Search space params: %s", list(self.search_space.keys()))

    def _resolve_model_config(self) -> Path:
        config_root = Path("src/configs")
        for directory in MODEL_CONFIG_DIRS:
            candidate = config_root / directory / f"{self.model_name.lower()}.yaml"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Config file for model '{self.model_name}' not found in src/configs/**"
        )

    def _build_base_config(self) -> Dict:
        with open(self.model_config_path, "r", encoding="utf-8") as fp:
            model_config = yaml.safe_load(fp)

        config = {**model_config, **self.project_config}
        config["model"] = self.model_name
        config["dataset"] = self.dataset
        data_path = Path(self.project_config["data_path"]).resolve()
        config["data_path"] = str(data_path)
        config["checkpoint_dir"] = str((self.output_dir / "checkpoints").resolve())
        config["show_progress"] = False
        config["log_wandb"] = False

        return config

    def _load_search_space(self) -> Dict[str, object]:
        space_path = Path("src/configs/tuning") / f"{self.model_name.lower()}_space.yaml"
        if not space_path.exists():
            self.logger.warning(
                "Search space file not found for %s. Using default learning_rate only.",
                self.model_name,
            )
            return DEFAULT_SPACE

        with open(space_path, "r", encoding="utf-8") as fp:
            raw_space = yaml.safe_load(fp) or {}

        search_space = {}
        for param, cfg in raw_space.items():
            ptype = cfg.get("type")
            label = f"{self.model_name}_{param}"
            if ptype == "choice":
                search_space[param] = hp.choice(label, cfg["values"])
            elif ptype == "uniform":
                search_space[param] = hp.uniform(label, cfg["min"], cfg["max"])
            elif ptype == "loguniform":
                search_space[param] = hp.loguniform(
                    label, np.log(cfg["min"]), np.log(cfg["max"])
                )
            elif ptype == "randint":
                search_space[param] = hp.randint(label, cfg["min"], cfg["max"])
            elif ptype == "quniform":
                search_space[param] = hp.quniform(label, cfg["min"], cfg["max"], cfg["q"])
            else:
                raise ValueError(f"Unsupported search space type '{ptype}' for param '{param}'")
        if not search_space:
            self.logger.warning(
                "Search space in %s is empty. Falling back to default learning_rate only.",
                space_path,
            )
            return DEFAULT_SPACE
        return search_space

    def _write_fixed_config(self) -> Path:
        temp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            yaml.safe_dump(self.base_config, temp, sort_keys=False)
        finally:
            temp.close()
        self.fixed_config_path = Path(temp.name)
        return self.fixed_config_path

    def _cleanup_fixed_config(self) -> None:
        if self.fixed_config_path and self.fixed_config_path.exists():
            self.fixed_config_path.unlink()
            self.fixed_config_path = None

    def run(self) -> Dict:
        self.logger.info(
            "Starting tuning | algo=%s | trials=%s | early_stop=%s",
            self.algo,
            self.max_evals,
            self.early_stop,
        )
        fixed_config_file = self._write_fixed_config()
        try:
            tuner = HyperTuning(
                objective_function=objective_function,
                space=self.search_space,
                fixed_config_file_list=[str(fixed_config_file)],
                algo=self.algo,
                max_evals=self.max_evals,
                early_stop=self.early_stop,
            )

            best_result, best_params = tuner.run()

            export_prefix = self.output_dir / f"{self.model_name.lower()}_{self.timestamp}"
            tuner.export_result(str(export_prefix))

            final_config = {**self.base_config, **best_params}
            best_config_path = self.output_dir / f"{self.model_name.lower()}_best_config.yaml"
            with open(best_config_path, "w", encoding="utf-8") as fp:
                yaml.safe_dump(final_config, fp, sort_keys=False)

            summary_path = self.output_dir / "best_result.yaml"
            with open(summary_path, "w", encoding="utf-8") as fp:
                yaml.safe_dump(
                    {
                        "model": self.model_name,
                        "dataset": self.dataset,
                        "algo": self.algo,
                        "max_evals": self.max_evals,
                        "early_stop": self.early_stop,
                        "best_result": best_result,
                        "best_params": best_params,
                    },
                    fp,
                    sort_keys=False,
                )

            self.logger.info("Best result: %s", best_result)
            self.logger.info("Best params: %s", best_params)
            self.logger.info("Best config saved to %s", best_config_path)

            return best_result
        finally:
            self._cleanup_fixed_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RecBole HyperTuning using project defaults",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Nome do modelo (ex.: GRU4Rec)")
    parser.add_argument("--dataset", help="Dataset (default: config/project_config.yaml)")
    parser.add_argument(
        "--algo",
        choices=["exhaustive", "random", "bayes"],
        default="random",
        help="Algoritmo de busca",
    )
    parser.add_argument("--max-evals", type=int, default=20, help="Número máximo de trials")
    parser.add_argument("--early-stop", type=int, default=10, help="Paciencia para early stop")
    parser.add_argument(
        "--output",
        help="Diretório customizado para salvar resultados (default: outputs/results/tuning/<ts>)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    tuner = RecBoleHyperparameterTuner(
        model_name=args.model,
        dataset=args.dataset,
        algo=args.algo,
        max_evals=args.max_evals,
        early_stop=args.early_stop,
        output_dir=args.output,
    )
    tuner.run()


if __name__ == "__main__":
    main()
