# Assignment 4 · Edge Computing & Intelligent Systems

Question 1 compares a ResNet-50 teacher with independently trained and distilled MobileNetV2, SqueezeNet 1.1, and EfficientNet-Lite0 students on CIFAR-100. Question 2 evolves CNNs on CIFAR-10 under measured ONNX CPU latency and storage limits, then trains the winning architecture from fresh weights.

## Contents

| Deliverable | Location |
| --- | --- |
| ResNet-50 and the three student networks | `edge/models.py`, `edge/train.py`, `models/teacher.pt` |
| Three baselines and three distilled models | `models/*_baseline.pt`, `models/*_kd.pt` |
| Hard-label, soft-target and feature distillation | `Distillation` in `edge/train.py` |
| CNN search space and evolutionary search | `edge/space.py`, `edge/nas.py` |
| NAS history, selected configuration and retrained model | `results/nas_search.{csv,json}`, `results/nas_best.json`, `models/nas_best.pt` |
| Export and CPU inference | `edge/deploy.py`, `models/*.onnx` (six students and the NAS model) |
| Benchmarking and numerical parity | `edge/benchmark.py`, `results/benchmark.csv`, `results/benchmark_environment.json`, `results/onnx_parity.csv` |
| Training curves and test metrics | `results/*_history.csv`, `results/*_metrics.json` |
| Technical report with execution screenshots | `Technical_Report.pdf` |

## 1. Set up

Use Python 3.11 or 3.12, a terminal opened in this directory, and about 1 GB of free space for datasets and generated artifacts, plus space for dependencies. A CUDA-compatible GPU speeds up full-data training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA training, install the matching PyTorch and TorchVision wheels using the [PyTorch installer](https://pytorch.org/get-started/previous-versions/) before installing the remaining requirements. The inference benchmark uses ONNX Runtime's CPU execution provider.

Training and dataset-index inference fetch CIFAR on first use. If the dataset host is slow, download the original Python archives from the [CIFAR project](https://cave.cs.toronto.edu/kriz/cifar.html) or its [Zenodo mirror](https://zenodo.org/records/10089977), then extract both into `data/`. TorchVision checks the extracted files. The official archive MD5 checksums are:

```text
c58f30108f718f92721af3b95e74349a  cifar-10-python.tar.gz
eb9058c3a382ffc7106e4002c42a8d85  cifar-100-python.tar.gz
```

## 2. Run predictions and benchmarks with the bundled weights

```bash
python -m edge.deploy infer --model models/mobilenetv2_kd.onnx --index 17
python -m edge.deploy infer --model models/nas_best.onnx --index 0
python -m edge.deploy infer --model models/squeezenet_baseline.pt --image /path/to/rgb_image.png
python -m edge.benchmark --threads 1 --warmup 20 --repeats 100
```

Change `--model` to any included student or NAS checkpoint or ONNX graph. With `--index`, inference prints the official CIFAR test label with the predictions. With `--image`, it resizes and normalizes the supplied RGB image and reports the top classes from the model's dataset. For local images, the ONNX backend needs only NumPy, Pillow and ONNX Runtime. The tensor shape is NCHW and the ONNX batch dimension is dynamic. The export checker compares ONNX and PyTorch for a batch of three.

Benchmarking writes `results/benchmark.csv` and `results/benchmark_environment.json`. It measures the ONNX file length and times batch-one FP32 execution with a preprocessed tensor, excluding preprocessing and session startup from latency. On supported systems, memory is the lifetime process peak RSS in an isolated subprocess and includes the interpreter and runtime. Peak growth is relative to the process high-water mark before opening the ONNX graph.

## 3. Reproduce the bundled experiments

Use an alternate output directory to preserve the bundled results. The seed is 42 for model training and search, and 71 for the stratified split. In Q1, each bundled training run uses 4,000 examples (40 per class), with 5,000 validation and 10,000 test examples. Baselines and their KD counterparts share architecture, seed, data split, augmentation, and training schedule. The bundled CPU run prioritizes testing the complete implementation and supports a comparable baseline/KD experiment.

```bash
python -m edge.train --mode teacher --device cpu --epochs 6 --train-limit 4000 --batch-size 64 --threads 4 --output runs/reproduction
python -m edge.train --mode baseline --device cpu --epochs 6 --train-limit 4000 --batch-size 64 --threads 3 --output runs/reproduction
python -m edge.train --mode kd --device cpu --epochs 6 --train-limit 4000 --batch-size 64 --threads 3 --output runs/reproduction
```

Use `--student mobilenetv2`, `--student squeezenet`, or `--student efficientnet_lite0` to train one student. The KD loss combines cross-entropy, temperature-scaled KL divergence, and feature-map alignment. To train a logit-only distilled variant, set `--feature-weight 0` and choose a different output directory. Use `--train-limit 0` for all 45,000 training examples and increase `--epochs` for the teacher and both student methods for a longer run.

The NAS experiment trains feasible candidates on a balanced 5,000-image proxy split. Candidate ranking uses the held-out 5,000-image validation split. The selected architecture is instantiated again, optimized from scratch on the full 45,000-image training split, and evaluated on the 10,000-image official test set.

```bash
python -m edge.nas --device cpu --threads 2 --final-threads 3 --batch-size 128 --trials 12 --population 4 \
  --proxy-epochs 3 --proxy-train-limit 5000 --final-epochs 6 --final-train-limit 0 \
  --max-p50-ms 2 --max-kib 128 --output runs/reproduction
python -m edge.deploy export --root runs/reproduction
python -m edge.benchmark --root runs/reproduction
```

NAS profiles every unique configuration on ONNX Runtime before proxy training. Candidates exceeding the median-latency or serialized ONNX storage ceilings are rejected and written to the search history. Supply `--max-rss-mib` to enforce a process RSS ceiling. Final export is profiled again after retraining. Use `--resume-final path/to/nas_best.json` with a fresh output directory to repeat only the final training from random initialization. Latency and borderline budget decisions can vary with CPU and system load.

## 4. Prepare the submission

Extract the ZIP and push the contents of this directory to your own GitHub repository. The ZIP contains the requested checkpoints and ONNX graphs, alongside the source, results and documentation. Use your own roll number and section in the final ZIP name, following `ROLL-NUM_SECTION.zip`. Submit it through Google Classroom.

## References

- [CIFAR-10 and CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html)
- [PyTorch knowledge distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- [TorchVision model builders](https://docs.pytorch.org/vision/stable/models.html)
- [EfficientNet-Lite0 in timm](https://huggingface.co/docs/timm/models/efficientnet)
- [ONNX Runtime Python documentation](https://onnxruntime.ai/docs/api/python/api_summary.html)
