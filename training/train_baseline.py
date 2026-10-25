"""Train a baseline with a module:function component factory."""
import argparse
import importlib
import json
from pathlib import Path
from baselines.source import verify_source
from baselines.adapters import NativeTrainingAdapter
from .data import TrainingBatches
from .loop import train
from .seed import set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--dataset-config", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--factory", required=True, help="Importable module:function for native training components")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    config = json.loads(args.dataset_config.read_text(encoding="utf-8"))
    if recipe.get("method_id") != args.method:
        p.error("Method and recipe differ")
    verify_source(args.method, args.source_dir)
    set_seed(0)
    module, function = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module), function)
    components = factory(config=config, source_dir=args.source_dir.resolve(), device=args.device)
    if not isinstance(components, dict) or "native" not in components:
        raise TypeError("Factory must return native and optional criterion components")
    model = NativeTrainingAdapter(args.method, components["native"], components.get("criterion"))
    batches = TrainingBatches(args.manifest, config, seed=0)
    summary = train(model, batches, recipe, args.output, device=args.device)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
