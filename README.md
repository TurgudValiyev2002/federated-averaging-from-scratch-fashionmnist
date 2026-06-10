# Federated Averaging from Scratch: FashionMNIST-Style Proxy

![Project overview](assets/readme_project_overview.png)

Figure: from-scratch federated averaging pipeline for multiclass classification.


## Motivation

Federated Averaging is easier to understand when we implement the training loop ourselves. This project builds the main FedAvg steps with NumPy rather than hiding them inside a deep learning framework.

## Project Goal

We implemented multiclass logistic regression from scratch and trained it with Federated Averaging across multiple clients.

## Dataset

We used the scikit-learn digits dataset as a small FashionMNIST-style image classification proxy. It is not FashionMNIST, but it allows the federated training logic to run locally without downloads.

## Tools

Python, NumPy, pandas, scikit-learn, and matplotlib.

## Method

The training data was split across six clients. Each client ran local gradient descent on its data. The server averaged client weights by client dataset size after each round.

## Hyperparameters

- Clients: 6
- Federated rounds: 7
- Local epochs per round: 2
- Learning rate: 0.2
- Test split: 20 percent
- Model: multiclass logistic regression from scratch

## Results

| Round | Test Accuracy |
|---:|---:|
| 1 | 0.8306 |
| 2 | 0.8417 |
| 3 | 0.8583 |
| 4 | 0.8722 |
| 5 | 0.8833 |
| 6 | 0.8917 |
| 7 | 0.9000 |

Results are saved in `results/fedavg_metrics.csv`, `results/client_sizes.csv`, and `results/accuracy_by_round.png`.

## Interpretation

Accuracy improved steadily from round 1 to round 7. This shows that the global model benefits from repeated local training and server averaging under these settings.

## Conclusion

The project demonstrates FedAvg from scratch. The next step should use real FashionMNIST and compare IID versus non-IID client splits.

## How To Run

```bash
pip install -r requirements.txt
python 1_fedavg_from_scratch.py
```
