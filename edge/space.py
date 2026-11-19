"""Architecture generator and mutation operators for evolutionary NAS."""
from copy import deepcopy

from torch import nn

RESOLUTIONS = (16, 24, 32)
CHANNELS = (12, 16, 24, 32, 48, 64)
KERNELS = (3, 5)
OPERATIONS = ("conv", "separable")
ACTIVATIONS = {"relu": nn.ReLU, "relu6": nn.ReLU6, "silu": nn.SiLU}
DEPTHS = (2, 3, 4, 5, 6)


def sample_block(rng):
    return {"channels": rng.choice(CHANNELS), "kernel": rng.choice(KERNELS),
            "op": rng.choice(OPERATIONS), "activation": rng.choice(tuple(ACTIVATIONS))}


def sample_config(rng):
    return {"resolution": rng.choice(RESOLUTIONS),
            "blocks": [sample_block(rng) for _ in range(rng.choice(DEPTHS))]}


def mutate(parent, rng):
    child = deepcopy(parent)
    blocks = child["blocks"]
    operation = rng.choice(("insert", "delete", "block", "resolution"))
    if operation == "insert" and len(blocks) < max(DEPTHS):
        blocks.insert(rng.randrange(len(blocks) + 1), sample_block(rng))
    elif operation == "delete" and len(blocks) > min(DEPTHS):
        blocks.pop(rng.randrange(len(blocks)))
    elif operation == "resolution":
        child["resolution"] = rng.choice([r for r in RESOLUTIONS if r != child["resolution"]])
    else:
        index = rng.randrange(len(blocks))
        key = rng.choice(tuple(blocks[index]))
        choices = {"channels": CHANNELS, "kernel": KERNELS,
                   "op": OPERATIONS, "activation": tuple(ACTIVATIONS)}
        blocks[index][key] = rng.choice([v for v in choices[key] if v != blocks[index][key]])
    return child


class SearchCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        blocks = config["blocks"]
        if config["resolution"] not in RESOLUTIONS or len(blocks) not in DEPTHS:
            raise ValueError("Invalid resolution or block count.")
        layers = []
        incoming = 3
        for index, gene in enumerate(blocks):
            width, kernel = gene["channels"], gene["kernel"]
            if width not in CHANNELS or kernel not in KERNELS or gene["op"] not in OPERATIONS:
                raise ValueError(f"Invalid NAS block: {gene}")
            activation = ACTIVATIONS[gene["activation"]]
            stride = 2 if index in (1, 3) else 1
            if gene["op"] == "separable":
                layers += [
                    nn.Conv2d(incoming, incoming, kernel, stride, kernel // 2,
                              groups=incoming, bias=False),
                    nn.BatchNorm2d(incoming),
                    activation(),
                    nn.Conv2d(incoming, width, 1, bias=False),
                ]
            else:
                layers.append(nn.Conv2d(incoming, width, kernel, stride, kernel // 2,
                                        bias=False))
            layers += [nn.BatchNorm2d(width), activation()]
            incoming = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(incoming, 10)

    def forward(self, images):
        return self.classifier(self.features(images).mean(dim=(2, 3)))
