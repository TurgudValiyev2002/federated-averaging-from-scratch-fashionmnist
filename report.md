# One-Page Report: Real FashionMNIST FedAvg

## Motivation

We use real FashionMNIST and compare IID versus non-IID client splits.

## Dataset

We used 6,000 FashionMNIST training images and 2,000 test images across 10 classes. Images are 28x28 grayscale.

## Method

We implemented multiclass logistic regression and FedAvg from scratch. Six clients trained locally for two epochs per round. The server averaged client weights by sample count.

## Results

In the IID split, test accuracy improved from 0.4400 in round 1 to 0.6855 in round 10. In the non-IID split, accuracy started at 0.1000 and reached 0.6470 by round 10.

## Interpretation

Non-IID training is harder because each client sees only part of the label space. This causes biased local updates and slower global learning. FedAvg still improves, but the non-IID curve is less stable.

## Conclusion

The project uses real FashionMNIST and demonstrates the central federated learning problem: client data heterogeneity changes training behavior.
