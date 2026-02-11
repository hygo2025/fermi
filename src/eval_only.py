import argparse
import json
import os
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml

import scipy.sparse as sp

if not hasattr(sp.dok_matrix, "_update"):
    def _update(self, data_dict):
        for k, v in data_dict.items():
            self[k] = v

    sp.dok_matrix._update = _update

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


MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def _parse_checkpoint_timestamp(checkpoint_path: Path) -> Optional[datetime]:
    stem = checkpoint_path.stem
    match = re.search(
        r"-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})$",
        stem,
    )
    if not match:
        return None
    mon, day, year, hour, minute, second = match.groups()
    return datetime(
        int(year),
        MONTH_MAP[mon],
        int(day),
        int(hour),
        int(minute),
        int(second),
    )


def _parse_model_name(checkpoint_path: Path) -> str:
    return checkpoint_path.stem.split("-", 1)[0]


def _parse_month_range_from_dir(exp_dir: Path) -> Optional[Tuple[int, int]]:
    match = re.match(r"^(\d{2})-(\d{2})$", exp_dir.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_run_time_from_dir(run_dir: Path) -> Optional[datetime]:
    match = re.search(r"run-(\d{8})_(\d{6})", run_dir.name)
    if not match:
        return None
    date_str, time_str = match.groups()
    return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")


def _load_wandb_config(path: Path) -> Tuple[dict, dict]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    external = raw.get("external_config_dict", {}).get("value", {})
    flat = {
        "model": raw.get("model", {}).get("value", external.get("model")),
        "dataset": raw.get("dataset", {}).get("value", external.get("dataset")),
        "external": external,
        "raw": raw,
    }
    return external, flat


def _load_wandb_config_remote(run) -> Tuple[dict, dict]:
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    external = config.get("external_config_dict", {})
    if isinstance(external, dict) and "value" in external:
        external = external["value"]
    if not external:
        raise ValueError("external_config_dict not found in remote run config")

    flat = {
        "model": config.get("model", external.get("model")),
        "dataset": config.get("dataset", external.get("dataset")),
        "external": external,
        "raw": config,
    }
    return external, flat


def _normalize_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"sim", "yes", "true", "1"}
    return False


def _resolve_wandb_config(
    checkpoint_path: Path,
    exp_dir: Path,
    wandb_dir: Path,
) -> Optional[Path]:
    model = _parse_model_name(checkpoint_path)
    ckpt_time = _parse_checkpoint_timestamp(checkpoint_path)
    month_range = _parse_month_range_from_dir(exp_dir)

    candidates = []
    for cfg_path in wandb_dir.rglob("config.yaml"):
        external, meta = _load_wandb_config(cfg_path)
        if meta["model"] != model:
            continue
        start_date = external.get("data_preparation", {}).get("start_date")
        end_date = external.get("data_preparation", {}).get("end_date")
        if month_range and start_date and end_date:
            start_month = int(start_date.split("-")[1])
            end_month = int(end_date.split("-")[1])
            if (start_month, end_month) != month_range:
                continue

        run_time = _parse_run_time_from_dir(cfg_path.parent.parent)
        if run_time and ckpt_time:
            delta = abs((run_time - ckpt_time).total_seconds())
        else:
            delta = float("inf")
        candidates.append((delta, run_time, cfg_path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1] or datetime.min))
    return candidates[0][2]


def _parse_wandb_ref(wandb_ref: str, default_project: str) -> Tuple[str, str]:
    if wandb_ref.startswith("wandb://"):
        tail = wandb_ref[len("wandb://"):]
        parts = tail.split("/")
        if len(parts) < 3:
            raise ValueError(
                "wandb ref must be wandb://<entity>/<project>/<run_id_or_name>"
            )
        project = "/".join(parts[:2])
        run_ref = "/".join(parts[2:])
        return project, run_ref

    if wandb_ref.startswith("wandb:"):
        return default_project, wandb_ref[len("wandb:"):]

    raise ValueError(f"Unsupported wandb ref: {wandb_ref}")


def _resolve_wandb_run(api, project: str, run_ref: str):
    try:
        return api.run(f"{project}/{run_ref}")
    except Exception:
        pass

    runs = api.runs(project, filters={"display_name": run_ref})
    for run in runs:
        if run.name == run_ref:
            return run

    # Fallback: scan for name match
    runs = api.runs(project)
    for run in runs:
        if run.name == run_ref:
            return run
    return None


def _ensure_data_path(
    data_path: Path,
    dataset: str,
    exp_dir: Path,
    inter_path: Optional[Path] = None,
) -> Path:
    dataset_inter = data_path / f"{dataset}.inter"
    if dataset_inter.exists():
        return data_path

    source_inter = inter_path if inter_path is not None else exp_dir / f"{dataset}.inter"
    if not source_inter.exists():
        raise FileNotFoundError(f"Missing .inter file at {source_inter}")

    stage_root = Path("/tmp") / "fermi_eval_data" / exp_dir.name
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_inter = stage_root / f"{dataset}.inter"
    if not stage_inter.exists():
        stage_inter.symlink_to(source_inter)
    return stage_root


def _json_friendly(result: dict) -> dict:
    cleaned = {}
    for k, v in result.items():
        if hasattr(v, "item"):
            cleaned[k] = float(v.item())
        else:
            cleaned[k] = v
    return cleaned


def _evaluate_once(
    model_name: str,
    checkpoint_path: Path,
    base_config: dict,
    dataset_name: str,
    eval_mode: str,
    eval_batch_size: Optional[int],
    device: Optional[str],
    wandb_group: str,
    disable_wandb: bool,
):
    config_dict = deepcopy(base_config)
    config_dict["dataset"] = dataset_name
    config_dict["show_progress"] = True

    if eval_batch_size is not None:
        config_dict["eval_batch_size"] = eval_batch_size

    if device is not None:
        config_dict["device"] = device

    if disable_wandb:
        config_dict["log_wandb"] = False

    config_dict["eval_args"] = deepcopy(config_dict.get("eval_args", {}))
    mode_cfg = config_dict["eval_args"].get("mode", {})
    if isinstance(mode_cfg, str):
        mode_cfg = {"test": mode_cfg}
    config_dict["eval_args"]["mode"] = mode_cfg
    if eval_mode == "full":
        print(config_dict["eval_args"])
        config_dict["eval_args"]["mode"]["test"] = "full"
    elif eval_mode == "uni100":
        config_dict["eval_args"]["mode"]["test"] = "uni100"
        config_dict.setdefault("eval_neg_sample_args", {})
        config_dict["eval_neg_sample_args"].setdefault("sample_num", 100)
        config_dict["eval_neg_sample_args"].setdefault("distribution", "uniform")
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    wandb_run = None
    if config_dict.get("log_wandb", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_tag = checkpoint_path.stem
        run_name = f"Eval_{model_name}_{eval_mode}_{ckpt_tag}_{timestamp}"
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            wandb_run = wandb.init(
                project=config_dict.get("wandb_project"),
                name=run_name,
                group=wandb_group,
                config=config_dict,
                reinit=True,
            )
        except Exception:
            wandb_run = None
        config_dict["log_wandb"] = False

    log(f"Evaluating checkpoint: {checkpoint_path}")
    log(f"Model: {model_name} | Dataset: {dataset_name} | Mode: {eval_mode}")
    if eval_batch_size is not None:
        log(f"Eval batch size override: {eval_batch_size}")
    if device is not None:
        log(f"Device override: {device}")

    log(f"Data path: {config_dict.get('data_path')}")
    log(f"Dataset save path: {config_dict.get('dataset_save_path')}")
    config = Config(model=model_name, config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    from recbole.utils import get_model
    model = get_model(model_name)(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    result = trainer.evaluate(
        test_data, model_file=str(checkpoint_path), show_progress=True
    )

    if wandb_run is not None:
        import wandb
        if wandb.run is not None:
            wandb.log({f"test_{k}": v for k, v in result.items()})
            wandb.finish()

    log(f"Resultados (test): {result}")
    return result


def _generate_queue(experiments_root: Path, months: list[str], output_path: Path):
    experiments = []
    for month in months:
        exp_dir = experiments_root / month
        if not exp_dir.exists():
            continue
        for checkpoint_path in sorted(exp_dir.glob("*.pth")):
            experiments.append(
                {
                    "id": f"{month}/{checkpoint_path.stem}",
                    "enabled": False,
                    "checkpoint": str(checkpoint_path),
                    "data_path": str(exp_dir),
                    "dataset": "realestate",
                    "model": _parse_model_name(checkpoint_path),
                    "inter_path": "",
                    "dataset_save_path": "",
                    "run_full": True,
                    "run_uni100": True,
                    "eval_batch_size": None,
                    "wandb_config": "",
                }
            )

    payload = {"experiments": experiments}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    log(f"Queue file generated: {output_path}")


def _run_batch(args):
    queue_path = Path(args.queue_file)
    with open(queue_path, "r") as f:
        queue = yaml.safe_load(f) or {}
    experiments = queue.get("experiments", [])
    if not experiments:
        log(f"No experiments found in queue file: {queue_path}")
        return

    wandb_dir = Path(args.wandb_dir)
    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = Path(args.done_file)
    done_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    if done_path.exists():
        with open(done_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("key")
                if key:
                    done_keys.add(key)

    for exp in experiments:
        if not _normalize_enabled(exp.get("enabled", False)):
            continue

        checkpoint_path = Path(exp["checkpoint"])
        exp_dir = Path(exp.get("data_path", checkpoint_path.parent))
        inter_path = exp.get("inter_path") or None
        inter_path = Path(inter_path) if inter_path else None
        dataset_save_path = exp.get("dataset_save_path") or None
        dataset_save_path = Path(dataset_save_path) if dataset_save_path else None
        exp_eval_batch_size = exp.get("eval_batch_size")
        dataset_name = exp.get("dataset", "realestate")
        model_name = exp.get("model") or _parse_model_name(checkpoint_path)

        wandb_config = exp.get("wandb_config")
        wandb_config_path = None
        base_config = None
        meta = None

        if wandb_config and str(wandb_config).startswith(("wandb:", "wandb://")):
            import wandb
            api = wandb.Api()
            project, run_ref = _parse_wandb_ref(str(wandb_config), args.wandb_project)
            run = _resolve_wandb_run(api, project, run_ref)
            if run is None:
                log(f"Skipping (remote run not found): {wandb_config}")
                continue
            base_config, meta = _load_wandb_config_remote(run)
        else:
            if wandb_config:
                wandb_config_path = Path(wandb_config)
            else:
                wandb_config_path = _resolve_wandb_config(
                    checkpoint_path, exp_dir, wandb_dir
                )
            if not wandb_config_path or not wandb_config_path.exists():
                log(f"Skipping (no wandb config found): {checkpoint_path}")
                continue
            base_config, meta = _load_wandb_config(wandb_config_path)
        if meta["model"]:
            model_name = meta["model"]
        if meta["dataset"]:
            dataset_name = meta["dataset"]

        if dataset_save_path and dataset_save_path.exists():
            base_config["dataset_save_path"] = str(dataset_save_path)
            base_config["save_dataset"] = False
        else:
            model_type = base_config.get("MODEL_TYPE") or meta.get("external", {}).get("MODEL_TYPE")
            if model_type:
                model_type = str(model_type).upper()
            else:
                model_type = None

            if model_type == "SEQUENTIAL":
                preferred = f"{dataset_name}-SequentialDataset.pth"
            elif model_type:
                preferred = f"{dataset_name}-Dataset.pth"
            else:
                preferred = None

            candidate = exp_dir / preferred if preferred else None
            if candidate and candidate.exists():
                base_config["dataset_save_path"] = str(candidate)
                base_config["save_dataset"] = False
                dataset_save_path = candidate
            else:
                # Heuristic fallback when MODEL_TYPE is missing
                seq_candidate = exp_dir / f"{dataset_name}-SequentialDataset.pth"
                base_candidate = exp_dir / f"{dataset_name}-Dataset.pth"
                sequential_models = {
                    "BERT4Rec",
                    "SASRec",
                    "GRU4Rec",
                    "NARM",
                    "Caser",
                    "NextItNet",
                    "TransRec",
                    "FPMC",
                    "FOSSIL",
                    "GCSAN",
                    "SRGNN",
                    "STAMP",
                    "HGN",
                }
                if model_name in sequential_models and seq_candidate.exists():
                    base_config["dataset_save_path"] = str(seq_candidate)
                    base_config["save_dataset"] = False
                    dataset_save_path = seq_candidate
                elif base_candidate.exists():
                    base_config["dataset_save_path"] = str(base_candidate)
                    base_config["save_dataset"] = False
                    dataset_save_path = base_candidate
                else:
                    data_path = _ensure_data_path(exp_dir, dataset_name, exp_dir, inter_path=inter_path)
                    base_config["data_path"] = str(data_path)

        exp_group = exp_dir.name.replace("-", "_")
        eval_group = f"eval_{exp_group}"
        run_full = exp.get("run_full", True)
        run_uni100 = exp.get("run_uni100", True)
        modes = []
        if _normalize_enabled(run_full):
            modes.append("full")
        if _normalize_enabled(run_uni100):
            modes.append("uni100")

        for mode in modes:
            key = f"{checkpoint_path}|{mode}"
            if key in done_keys:
                log(f"Skipping (already done): {checkpoint_path} | Mode: {mode}")
                continue
            result = _evaluate_once(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                base_config=base_config,
                dataset_name=dataset_name,
                eval_mode=mode,
                eval_batch_size=exp_eval_batch_size if exp_eval_batch_size is not None else args.eval_batch_size,
                device=args.device,
                wandb_group=eval_group,
                disable_wandb=args.no_wandb,
            )

            record = {
                "key": key,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "checkpoint": str(checkpoint_path),
                "experiment_dir": str(exp_dir),
                "model": model_name,
                "dataset": dataset_name,
                "eval_mode": mode,
                "inter_path": str(inter_path) if inter_path else "",
                "dataset_save_path": str(dataset_save_path) if dataset_save_path else "",
                "wandb_config": str(wandb_config_path) if wandb_config_path else str(wandb_config),
                "result": _json_friendly(result),
            }
            with open(results_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            with open(done_path, "a") as f:
                f.write(json.dumps({"key": key, "timestamp": record["timestamp"]}, ensure_ascii=True) + "\n")
            done_keys.add(key)


def _run_single(args):
    project_config = get_config()
    config_dict = load_model_config(args.model, project_config)

    dataset_name = args.dataset or project_config["dataset"]
    config_dict["dataset"] = dataset_name
    config_dict["show_progress"] = True

    if args.eval_batch_size is not None:
        config_dict["eval_batch_size"] = args.eval_batch_size

    if args.device is not None:
        config_dict["device"] = args.device

    if args.wandb_group:
        config_dict["wandb_group"] = args.wandb_group

    wandb_run = None
    if config_dict.get("log_wandb", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"Eval_{args.model}_{timestamp}"
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            wandb_run = wandb.init(
                project=config_dict.get("wandb_project"),
                name=run_name,
                group=config_dict.get("wandb_group"),
                config=config_dict,
                reinit=True,
            )
        except Exception:
            wandb_run = None
        config_dict["log_wandb"] = False

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

    if wandb_run is not None:
        import wandb
        if wandb.run is not None:
            wandb.log({f"test_{k}": v for k, v in result.items()})
            wandb.finish()

    log(f"Resultados (test): {result}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved RecBole checkpoints")
    parser.add_argument("--model", required=False, help="Model name (e.g., TransRec)")
    parser.add_argument("--checkpoint", required=False, help="Path to saved .pth checkpoint")
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
    parser.add_argument(
        "--queue-file",
        default=None,
        help="YAML queue file for batch evaluation",
    )
    parser.add_argument(
        "--generate-queue",
        action="store_true",
        help="Generate a queue YAML file and exit",
    )
    parser.add_argument(
        "--experiments-root",
        default="/home/hygo2025/Documents/experimentos/mg",
        help="Root directory with month folders (e.g., 03-04)",
    )
    parser.add_argument(
        "--months",
        default="03-04,04-05,05-06",
        help="Comma-separated month folders to include",
    )
    parser.add_argument(
        "--wandb-dir",
        default="/home/hygo2025/Development/projects/fermi/wandb",
        help="W&B directory containing run config.yaml files",
    )
    parser.add_argument(
        "--wandb-project",
        default="hygo2025-ufes/fermi",
        help="W&B project for remote run lookups (entity/project)",
    )
    parser.add_argument(
        "--results-file",
        default="outputs/results/eval_batch_results.jsonl",
        help="Path to write batch evaluation results (JSONL)",
    )
    parser.add_argument(
        "--done-file",
        default="outputs/results/eval_batch_done.jsonl",
        help="Path to track completed evaluations (JSONL)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging during evaluation",
    )
    args = parser.parse_args()

    if args.generate_queue:
        if not args.queue_file:
            raise ValueError("--queue-file is required with --generate-queue")
        months = [m.strip() for m in args.months.split(",") if m.strip()]
        _generate_queue(
            experiments_root=Path(args.experiments_root),
            months=months,
            output_path=Path(args.queue_file),
        )
        return

    if args.queue_file:
        _run_batch(args)
        return

    if not args.model or not args.checkpoint:
        raise ValueError("--model and --checkpoint are required for single evaluation")
    _run_single(args)


if __name__ == "__main__":
    main()
