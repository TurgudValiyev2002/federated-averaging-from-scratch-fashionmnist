from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results")
RNG = np.random.default_rng(42)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def local_train(x, y, w, b, classes, lr=0.2, epochs=1):
    w = w.copy()
    b = b.copy()
    y_one = np.eye(classes)[y]
    for _ in range(epochs):
        p = softmax(x @ w + b)
        grad_w = x.T @ (p - y_one) / len(x)
        grad_b = (p - y_one).mean(axis=0)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def main():
    RESULTS.mkdir(exist_ok=True)
    x, y = load_digits(return_X_y=True)
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    n_clients = 6
    client_indices = np.array_split(RNG.permutation(len(x_train)), n_clients)
    classes = len(np.unique(y))
    w = np.zeros((x_train.shape[1], classes))
    b = np.zeros(classes)
    rows = []
    for rnd in range(1, 8):
        local_ws, local_bs, sizes = [], [], []
        for idx in client_indices:
            lw, lb = local_train(x_train[idx], y_train[idx], w, b, classes, lr=0.2, epochs=2)
            local_ws.append(lw)
            local_bs.append(lb)
            sizes.append(len(idx))
        w = np.average(local_ws, axis=0, weights=sizes)
        b = np.average(local_bs, axis=0, weights=sizes)
        pred = (x_test @ w + b).argmax(axis=1)
        rows.append({"round": rnd, "test_accuracy": round(accuracy_score(y_test, pred), 4)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "fedavg_metrics.csv", index=False)
    pd.DataFrame({"client": range(n_clients), "train_samples": [len(i) for i in client_indices]}).to_csv(RESULTS / "client_sizes.csv", index=False)
    plt.figure(figsize=(6,4))
    plt.plot(metrics["round"], metrics["test_accuracy"], marker="o")
    plt.ylim(0,1)
    plt.xlabel("Federated round")
    plt.ylabel("Test accuracy")
    plt.title("FedAvg From Scratch")
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_by_round.png", dpi=160)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
