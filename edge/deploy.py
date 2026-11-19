"""Export PyTorch classifiers and run ONNX Runtime CPU inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def cpu_session(path, threads=1):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def export_model(model, spec, path):
    import onnx
    import torch

    model.eval()
    parameter = next(model.parameters())
    resolution = spec["resolution"]
    dummy = torch.zeros(1, 3, resolution, resolution, device=parameter.device)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.onnx")
    with torch.inference_mode():
        torch.onnx.export(model, dummy, temporary, input_names=["image"],
                          output_names=["logits"], opset_version=17, dynamo=False,
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    graph = onnx.load(temporary)
    onnx.helper.set_model_props(graph, {"edge_model": json.dumps(spec)})
    onnx.checker.check_model(graph)
    onnx.save(graph, temporary)
    temporary.replace(path)


def load_pytorch(path, threads=1):
    import torch
    from .models import build_model
    from .train import read_checkpoint

    torch.set_num_threads(threads)
    checkpoint = read_checkpoint(path)
    spec = checkpoint["spec"]
    model = build_model(spec)
    model.load_state_dict(checkpoint["state"])
    return model.eval(), spec


def export_checkpoints(args):
    from .train import write_csv
    import torch

    rows = []
    for checkpoint in args.root.glob("models/*.pt"):
        if checkpoint.stem == "teacher" and not args.include_teacher:
            continue
        model, spec = load_pytorch(checkpoint)
        path = checkpoint.with_suffix(".onnx")
        export_model(model, spec, path)
        session = cpu_session(path)
        with torch.inference_mode():
            size = spec["resolution"]
            tensor = torch.randn(3, 3, size, size, generator=torch.Generator().manual_seed(96))
            reference = model(tensor).numpy()
        actual = session.run(["logits"], {"image": tensor.numpy()})[0]
        np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=3e-4)
        np.testing.assert_array_equal(actual.argmax(1), reference.argmax(1))
        rows.append({"model": checkpoint.stem, "tested_batch": len(tensor),
                     "max_abs_logit_error": float(np.abs(actual - reference).max()),
                     "matching_top1": bool(np.array_equal(actual.argmax(1), reference.argmax(1)))})
        print(f"{path} | {path.stat().st_size:,} bytes | PyTorch/ONNX parity passed")
    if not rows:
        raise SystemExit(f"No deployable checkpoints found in {args.root / 'models'}")
    write_csv(args.root / "results" / "onnx_parity.csv", sorted(rows, key=lambda row: row["model"]))


def infer(args):
    from PIL import Image
    from .preprocess import prepare_image

    if args.model.suffix == ".onnx":
        session = cpu_session(args.model, args.threads)
        spec = json.loads(session.get_modelmeta().custom_metadata_map["edge_model"])
    elif args.model.suffix == ".pt":
        model, spec = load_pytorch(args.model, args.threads)
    else:
        raise SystemExit("Provide an .onnx or .pt model.")
    if args.threads < 1:
        raise SystemExit("Threads must be positive.")
    expected = None
    if args.image is not None:
        with Image.open(args.image) as file:
            image = file.convert("RGB")
    else:
        from .data import load_cifar
        _, testing = load_cifar(spec["dataset"], args.data)
        if not 0 <= args.index < len(testing):
            raise SystemExit(f"Test index must be 0–{len(testing) - 1}.")
        image = Image.fromarray(testing.data[args.index])
        expected = testing.classes[testing.targets[args.index]]
    inputs = prepare_image(image, spec["dataset"], spec["resolution"])
    if args.model.suffix == ".onnx":
        logits = session.run(["logits"], {"image": inputs})[0][0]
    else:
        import torch
        with torch.inference_mode():
            logits = model(torch.from_numpy(inputs)).numpy()[0]
    if not 1 <= args.top_k <= len(spec["labels"]):
        raise SystemExit("Top-k must be between 1 and the number of classes.")
    weights = np.exp(logits - logits.max())
    probabilities = weights / weights.sum()
    indexes = np.argsort(-probabilities, kind="stable")[:args.top_k]
    payload = {"model": spec["tag"], "backend": args.model.suffix[1:],
               "source": args.image.name if args.image is not None else
               f"{spec['dataset']} test #{args.index}",
               "expected": expected,
               "predictions": [{"label": spec["labels"][i], "probability": round(float(probabilities[i]), 6)}
                               for i in indexes]}
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tools = parser.add_subparsers(dest="command", required=True)
    exporter = tools.add_parser("export", help="Export and verify every available checkpoint")
    exporter.add_argument("--root", type=Path, default=Path("."))
    exporter.add_argument("--include-teacher", action="store_true")
    predictor = tools.add_parser("infer", help="Run one image through a checkpoint or ONNX file")
    predictor.add_argument("--model", type=Path, required=True)
    inputs = predictor.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path)
    inputs.add_argument("--index", type=int)
    predictor.add_argument("--data", type=Path, default=Path("data"))
    predictor.add_argument("--threads", type=int, default=1)
    predictor.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.command == "export":
        export_checkpoints(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
