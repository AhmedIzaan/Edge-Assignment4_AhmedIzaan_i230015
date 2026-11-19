"""Evolve CIFAR-10 CNNs against measured ONNX deployment constraints."""
import argparse
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from .benchmark import isolated_profile
from .data import load_cifar, make_loaders
from .deploy import export_model
from .models import build_model
from .space import sample_config, mutate
from .train import add_training_options, fit, init, write_csv, write_json


def grid_row(entry):
    row = {key: value for key, value in entry.items() if key != "config"}
    row.update(resolution=entry["config"]["resolution"], depth=len(entry["config"]["blocks"]),
               config_json=json.dumps(entry["config"], separators=(",", ":")))
    return row


def violations(resources, limits):
    exceeded = []
    for key, limit in limits.items():
        if limit is not None and (resources[key] is None or resources[key] > limit):
            exceeded.append(f"{key} unavailable" if resources[key] is None else key)
    return exceeded


def ranking(entry):
    return (entry["val_accuracy"], -entry["p50_ms"], -entry["storage_bytes"])


def retrain(args, data, summary):
    winner = summary["winner"]
    limits = summary["limits"]
    metadata = {"architecture": "nas", "dataset": "cifar10", "method": "evolution",
                "labels": data[0].classes}
    args.split_seed = summary["split_seed"]
    if args.final_threads is not None:
        args.threads = args.final_threads
    summary = {**summary, "final_epochs": args.final_epochs,
               "final_train_limit": args.final_train_limit, "final_seed": args.seed,
               "final_batch_size": args.batch_size, "final_lr": args.lr,
               "final_threads": args.threads}
    best_path = args.output / "results" / "nas_best.json"
    spec = {**metadata, "config": winner["config"], "resolution": winner["config"]["resolution"],
            "search_trial": winner["trial"]}
    print(f"Selected trial {winner['trial']} with validation accuracy "
          f"{winner['val_accuracy']:.2%}. Retraining from scratch.", flush=True)
    write_json(best_path, summary)
    loaders = make_loaders(spec, args, *data, args.final_train_limit)
    # fit() constructs a fresh model. The search trained and kept no candidate weights.
    model, trained = fit(spec, args, loaders, args.final_epochs, tag="nas_best")
    final_path = args.output / "models" / "nas_best.onnx"
    export_model(model, trained, final_path)
    measured = isolated_profile(final_path, trained["resolution"])
    rejected = violations(measured, limits)
    write_json(best_path, {**summary, "final_metrics": trained, "final_profile": measured,
                           "final_resource_violations": rejected})
    if rejected:
        raise SystemExit(f"Retrained model exceeded the measured budget: {', '.join(rejected)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_options(parser)
    parser.add_argument("--resume-final", type=Path,
                        help="Start fresh weights using an existing nas_best.json selection")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--proxy-epochs", type=int, default=4)
    parser.add_argument("--final-threads", type=int, help="Training threads for only final retraining")
    parser.add_argument("--final-epochs", type=int, default=35)
    parser.add_argument("--proxy-train-limit", type=int, default=10000)
    parser.add_argument("--final-train-limit", type=int, default=0)
    parser.add_argument("--max-p50-ms", type=float, default=2.0)
    parser.add_argument("--max-kib", type=float, default=320.0)
    parser.add_argument("--max-rss-mib", type=float)
    args = parser.parse_args()
    if (min(args.trials, args.population, args.proxy_epochs, args.final_epochs, args.threads) < 1
            or args.batch_size < 2 or min(args.max_p50_ms, args.max_kib) <= 0
            or min(args.proxy_train_limit, args.final_train_limit) < 0
            or (args.final_threads is not None and args.final_threads < 1)
            or (args.max_rss_mib is not None and args.max_rss_mib <= 0)):
        parser.error("Invalid search options. See --help.")
    init(args.seed, args.threads)
    rng = random.Random(args.seed)
    data = load_cifar("cifar10", args.data)
    if args.resume_final is not None:
        selection = json.loads(args.resume_final.read_text())
        search_fields = ("winner", "limits", "seed", "split_seed", "population", "trials",
                         "proxy_epochs", "proxy_train_limit")
        summary = {key: selection[key] for key in search_fields}
        if "search_threads" in selection:
            summary["search_threads"] = selection["search_threads"]
        retrain(args, data, summary)
        return
    results = args.output / "results"
    limits = {"p50_ms": args.max_p50_ms, "storage_bytes": int(args.max_kib * 1024),
              "peak_rss_mib": args.max_rss_mib}
    metadata = {"architecture": "nas", "dataset": "cifar10", "method": "evolution",
                "labels": data[0].classes}
    seen, history, pool = set(), [], []

    with TemporaryDirectory() as workspace:
        artifact = Path(workspace) / "candidate.onnx"
        for index in range(args.trials):
            parent = None
            for _ in range(100):
                parent = None
                if index >= args.population and pool and rng.random() < 0.85:
                    parent = max(rng.sample(pool, min(len(pool), 3)), key=ranking)
                    config = mutate(parent["config"], rng)
                else:
                    config = sample_config(rng)
                key = json.dumps(config, sort_keys=True)
                if key not in seen:
                    break
            else:
                raise RuntimeError("Could not sample a novel candidate.")
            seen.add(key)
            tag = f"candidate_{index + 1:03}"
            spec = {**metadata, "config": config, "resolution": config["resolution"]}
            # Deployment footprint and execution cost are measured before spending training time.
            export_model(build_model(spec), {**spec, "tag": tag}, artifact)
            metrics = isolated_profile(artifact, spec["resolution"])
            rejected = violations(metrics, limits)
            entry = {"trial": index + 1, "parent_trial": parent["trial"] if parent else None,
                     "config": config, **metrics, "accepted": not rejected,
                     "reason": ", ".join(rejected), "val_accuracy": None}
            if not rejected:
                loaders = make_loaders(spec, args, *data, args.proxy_train_limit, skip_test=True)
                _, proxy = fit(spec, args, loaders, args.proxy_epochs, tag=tag, save=False)
                entry["val_accuracy"] = proxy["val_accuracy"]
                pool = sorted([*pool, entry], key=ranking, reverse=True)[:args.population]
            decision = "accepted" if not rejected else f"rejected ({entry['reason']})"
            print(f"{tag} | {decision} | p50 {metrics['p50_ms']:.3f} ms | "
                  f"{metrics['storage_bytes']:,} B", flush=True)
            history.append(entry)
            write_csv(results / "nas_search.csv", [grid_row(row) for row in history])
            write_json(results / "nas_search.json", history)

    if not pool:
        raise SystemExit("No candidate passed the deployment budget. See results/nas_search.csv.")
    winner = max(pool, key=ranking)
    summary = {"winner": winner, "limits": limits, "seed": args.seed, "split_seed": args.split_seed,
               "population": args.population, "trials": args.trials, "proxy_epochs": args.proxy_epochs,
               "proxy_train_limit": args.proxy_train_limit, "search_threads": args.threads}
    retrain(args, data, summary)


if __name__ == "__main__":
    main()
