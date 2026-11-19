"""Train the ResNet-50 teacher and matched baseline/KD CIFAR-100 students."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_cifar, make_loaders
from .models import STUDENTS, build_model


def init(seed, threads):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def add_training_options(parser):
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=71)
    parser.add_argument("--lr", type=float, default=0.001)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class Distillation(nn.Module):
    def __init__(self, teacher, student, temperature, soft_weight, feature_weight):
        super().__init__()
        self.teacher = teacher.eval().requires_grad_(False)
        self.temperature = temperature
        self.soft_weight = soft_weight
        self.feature_weight = feature_weight
        self.project = nn.Conv2d(student.feature_channels, teacher.feature_channels, 1,
                                 bias=False) if feature_weight else None

    def forward(self, student, images, targets):
        if self.project is not None:
            student_logits, student_features = student.logits_and_features(images)
            with torch.no_grad():
                teacher_logits, teacher_features = self.teacher.logits_and_features(images)
            spatial_size = (2, 2)
            aligned_student = self.project(F.adaptive_avg_pool2d(student_features, spatial_size))
            aligned_teacher = F.adaptive_avg_pool2d(teacher_features, spatial_size)
            # Unit-normalization keeps feature scales comparable across architectures.
            dist = F.mse_loss(F.normalize(aligned_student, dim=1),
                              F.normalize(aligned_teacher, dim=1), reduction="sum")
            feature = dist / (len(targets) * spatial_size[0] * spatial_size[1])
        else:
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = self.teacher(images)
            feature = student_logits.new_zeros(())
        temperature = self.temperature
        hard = F.cross_entropy(student_logits, targets)
        soft = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                        F.softmax(teacher_logits / temperature, dim=1),
                        reduction="batchmean") * temperature ** 2
        combined = (1 - self.soft_weight) * hard + self.soft_weight * soft
        return student_logits, combined + self.feature_weight * feature, hard, soft, feature


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    size = len(loader.dataset)
    correct = loss = 0
    for images, targets in loader:
        images, targets = images.to(device, memory_format=torch.channels_last), targets.to(device)
        logits = model(images)
        correct += int((logits.argmax(1) == targets).sum())
        loss += F.cross_entropy(logits, targets, reduction="sum").item()
    return {"accuracy": correct / size, "loss": loss / size, "examples": size}


def fit(spec, args, loaders, epochs, *, tag, teacher=None, save=True):
    init(args.seed, args.threads)
    device = args.device
    model = build_model(spec).to(device, memory_format=torch.channels_last)
    train, val, test = loaders
    kd = None
    params = list(model.parameters())
    if teacher is not None:
        kd = Distillation(teacher, model, args.temperature, args.soft_weight,
                          args.feature_weight).to(device)
        if kd.project is not None:
            params.extend(kd.project.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy, best_state = -1, None
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(4)
        correct = seen = 0
        for images, targets in train:
            images = images.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            if kd is None:
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                parts = [loss.item(), loss.item(), 0., 0.]
            else:
                logits, loss, hard, soft, feature = kd(model, images, targets)
                parts = [tensor.item() for tensor in (loss, hard, soft, feature)]
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(1) == targets).sum())
            seen += len(targets)
            totals += np.array(parts) * len(targets)

        validation = evaluate(model, val, device)
        row = dict(zip(("train_loss", "train_ce", "train_kl_t2", "train_feature"), totals / seen))
        row.update(epoch=epoch, train_accuracy=correct / seen, val_accuracy=validation["accuracy"],
                   val_loss=validation["loss"], train_examples=seen)
        history.append(row)
        scheduler.step()
        print(f"{tag} | epoch {epoch}/{epochs} | loss {row['train_loss']:.4f} | "
              f"validation {validation['accuracy']:.2%}", flush=True)

        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    metadata = {**spec, "tag": tag, "best_epoch": best_epoch, "epochs": epochs,
                "seed": args.seed, "split_seed": args.split_seed, "lr": args.lr,
                "batch_size": args.batch_size, "train_examples": len(train.dataset),
                "val_examples": len(val.dataset), "val_accuracy": best_accuracy,
                "optimization_seconds": time.perf_counter() - started}
    if kd is not None:
        metadata["kd"] = {"temperature": args.temperature, "soft_weight": args.soft_weight,
                          "feature_weight": args.feature_weight, "teacher": "teacher"}
    if test is not None:
        metrics = evaluate(model, test, device)
        metadata.update({f"test_{key}": value for key, value in metrics.items()})
        print(f"{tag} | best epoch {best_epoch} | test {metrics['accuracy']:.2%}", flush=True)
    if save:
        model_path = args.output / "models" / f"{tag}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(".tmp")
        torch.save({"spec": metadata, "state": best_state}, temporary)
        temporary.replace(model_path)
        write_csv(args.output / "results" / f"{tag}_history.csv", history)
        write_json(args.output / "results" / f"{tag}_metrics.json", metadata)
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_options(parser)
    parser.add_argument("--mode", choices=("teacher", "baseline", "kd"), required=True)
    parser.add_argument("--student", choices=(*STUDENTS, "all"), default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--soft-weight", type=float, default=0.4)
    parser.add_argument("--feature-weight", type=float, default=0.15)
    parser.add_argument("--teacher", type=Path)
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 2 or args.threads < 1 or args.train_limit < 0
            or args.temperature <= 0 or not 0 <= args.soft_weight <= 1 or args.feature_weight < 0):
        parser.error("Invalid training options. See --help.")
    init(args.seed, args.threads)
    data = load_cifar("cifar100", args.data)
    teacher = None
    if args.mode == "kd":
        load = read_checkpoint(args.teacher or args.output / "models" / "teacher.pt")
        if load["spec"]["architecture"] != "resnet50":
            parser.error("KD expects a ResNet-50 teacher checkpoint.")
        teacher = build_model(load["spec"]).to(args.device, memory_format=torch.channels_last)
        teacher.load_state_dict(load["state"])
    architectures = ("resnet50",) if args.mode == "teacher" else (
        STUDENTS if args.student == "all" else (args.student,))
    for name in architectures:
        spec = {"architecture": name, "dataset": "cifar100", "resolution": 32,
                "labels": data[0].classes, "method": args.mode}
        loaders = make_loaders(spec, args, *data, args.train_limit)
        tag = "teacher" if args.mode == "teacher" else f"{name}_{args.mode}"
        fit(spec, args, loaders, args.epochs, tag=tag, teacher=teacher)


if __name__ == "__main__":
    main()
