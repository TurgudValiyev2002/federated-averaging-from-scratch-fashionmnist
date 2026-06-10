# Report: Federated Averaging from Scratch

## Motivation

We implemented FedAvg from scratch to understand the algorithm instead of only using a library.

## Dataset

The experiment used the scikit-learn digits dataset as a local image-classification proxy. It is not FashionMNIST.

## Method

Six clients trained local multiclass logistic regression models with gradient descent. The server averaged their parameters after each round.

## Hyperparameters

We used 6 clients, 7 rounds, 2 local epochs, learning rate 0.2, and a 20 percent test split.

## Results

Test accuracy improved from 0.8306 in round 1 to 0.9000 in round 7.

## Interpretation

The global model improved as clients repeatedly trained locally and shared averaged parameters. The steady increase shows that the simple FedAvg loop is working.

## Conclusion

The project shows the core FedAvg training loop clearly. A stronger version should use real FashionMNIST and non-IID partitions.
