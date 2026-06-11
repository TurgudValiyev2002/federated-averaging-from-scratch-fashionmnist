from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
SEED = 42


def download_fashionmnist() -> None:
    DATA.mkdir(exist_ok=True)
    for filename in FILES.values():
        path = DATA / filename
        if not path.exists():
            url = f"{BASE_URL}/{filename}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, n, rows, cols = struct.unpack(">IIII", handle.read(16))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, n = struct.unpack(">II", handle.read(8))
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_subset(train_per_class: int = 600, test_per_class: int = 200):
    download_fashionmnist()
    x_train_all = read_idx_images(DATA / FILES["train_images"])
    y_train_all = read_idx_labels(DATA / FILES["train_labels"])
    x_test_all = read_idx_images(DATA / FILES["test_images"])
    y_test_all = read_idx_labels(DATA / FILES["test_labels"])
    rng = np.random.default_rng(SEED)

    def balanced(x, y, per_class):
        indices = []
        for label in range(10):
            label_idx = np.where(y == label)[0]
            indices.extend(rng.choice(label_idx, size=per_class, replace=False))
        indices = np.array(indices)
        rng.shuffle(indices)
        return x[indices], y[indices]

    return (*balanced(x_train_all, y_train_all, train_per_class), *balanced(x_test_all, y_test_all, test_per_class))


def one_hot(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def local_train(w: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray, epochs: int, lr: float):
    y_oh = one_hot(y, w.shape[1])
    for _ in range(epochs):
        probs = softmax(x @ w + b)
        grad_w = x.T @ (probs - y_oh) / len(x)
        grad_b = (probs - y_oh).mean(axis=0)
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, b


def evaluate(w: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    pred = (x @ w + b).argmax(axis=1)
    return accuracy_score(y, pred)


def make_clients_iid(y: np.ndarray, clients: int = 6) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    return [idx for idx in np.array_split(rng.permutation(len(y)), clients)]


def make_clients_noniid(y: np.ndarray, clients: int = 6) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    class_groups = {label: rng.permutation(np.where(y == label)[0]).tolist() for label in range(10)}
    assignments = [[0, 1], [2, 3], [4, 6], [5, 7], [8, 9], [0, 2, 4, 6, 8]]
    client_indices = []
    for labels in assignments:
        selected = []
        for label in labels:
            take = min(220, len(class_groups[label]))
            selected.extend(class_groups[label][:take])
            class_groups[label] = class_groups[label][take:]
        client_indices.append(np.array(selected, dtype=int))
    return client_indices


def run_fedavg(x_train, y_train, x_test, y_test, split_name: str, client_indices: list[np.ndarray]):
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.01, (x_train.shape[1], 10)).astype(np.float32)
    b = np.zeros(10, dtype=np.float32)
    rows = []
    rounds = 10
    local_epochs = 2
    lr = 0.35
    for rnd in range(1, rounds + 1):
        local_ws, local_bs, weights = [], [], []
        for idx in client_indices:
            lw, lb = local_train(w.copy(), b.copy(), x_train[idx], y_train[idx], local_epochs, lr)
            local_ws.append(lw)
            local_bs.append(lb)
            weights.append(len(idx))
        w = np.average(local_ws, axis=0, weights=weights)
        b = np.average(local_bs, axis=0, weights=weights)
        rows.append({"split": split_name, "round": rnd, "test_accuracy": round(evaluate(w, b, x_test, y_test), 4)})
    clients = pd.DataFrame(
        [
            {
                "split": split_name,
                "client": cid,
                "train_samples": len(idx),
                "labels": " ".join(str(v) for v in sorted(set(y_train[idx]))),
            }
            for cid, idx in enumerate(client_indices)
        ]
    )
    return pd.DataFrame(rows), clients


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test = load_subset()
    iid_metrics, iid_clients = run_fedavg(x_train, y_train, x_test, y_test, "iid", make_clients_iid(y_train))
    noniid_metrics, noniid_clients = run_fedavg(x_train, y_train, x_test, y_test, "non_iid", make_clients_noniid(y_train))
    metrics = pd.concat([iid_metrics, noniid_metrics], ignore_index=True)
    clients = pd.concat([iid_clients, noniid_clients], ignore_index=True)
    metrics.to_csv(RESULTS / "fedavg_metrics.csv", index=False)
    clients.to_csv(RESULTS / "client_sizes.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "train_images", "value": len(x_train)},
            {"metric": "test_images", "value": len(x_test)},
            {"metric": "classes", "value": 10},
            {"metric": "clients", "value": 6},
            {"metric": "rounds", "value": 10},
            {"metric": "local_epochs", "value": 2},
            {"metric": "learning_rate", "value": 0.35},
        ]
    ).to_csv(RESULTS / "experiment_setup.csv", index=False)

    plt.figure(figsize=(7, 4))
    for split, group in metrics.groupby("split"):
        plt.plot(group["round"], group["test_accuracy"], marker="o", label=split)
    plt.ylim(0, 1)
    plt.xlabel("Federated round")
    plt.ylabel("Test accuracy")
    plt.title("Real FashionMNIST FedAvg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_by_round.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for label, ax in enumerate(axes.ravel()):
        image = x_test[y_test == label][0].reshape(28, 28)
        ax.imshow(image, cmap="gray")
        ax.set_title(CLASS_NAMES[label], fontsize=8)
        ax.axis("off")
    fig.suptitle("Real FashionMNIST examples", y=0.98)
    fig.tight_layout()
    fig.savefig(RESULTS / "fashionmnist_examples.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("FashionMNIST\nreal images", 0.15),
        ("6 clients\nIID/non-IID", 0.40),
        ("Local logistic\ntraining", 0.64),
        ("Weighted\nFedAvg", 0.86),
    ]
    for text, xpos in boxes:
        ax.text(xpos, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.11, 0.55), xytext=(start[1] + 0.11, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Real FashionMNIST FedAvg workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "readme_project_overview.png", dpi=180)
    plt.close(fig)

    print(metrics.tail(4).to_string(index=False))


if __name__ == "__main__":
    main()
