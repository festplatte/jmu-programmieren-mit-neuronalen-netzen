# NN-Framework

Folgende Netze wurden trainiert:

## Fully-Connected-Network

```kotlin
val fcNetwork = Network(ImageStringInputLayer(), listOf(
        FlatternLayer(Shape(intArrayOf(1, 784))),
        FullyConnectedLayer(Shape(intArrayOf(1, 784)), Shape(intArrayOf(1, 512))),
        SigmoidActivation(Shape(intArrayOf(1, 512))),
        FullyConnectedLayer(Shape(intArrayOf(1, 512)), Shape(intArrayOf(1, 10))),
        SoftmaxLayer(Shape(intArrayOf(1, 10)))
))
```
mit folgenden Hyperparametern:

- Batchsize: 256
- Learning-Rate: 0,001
- Epochen: 5
- Loss: Cross-Entropy
- Optimizer: Stochastic Gradient Descent

Ergebnis: `Validation finished - Loss: 0.30303422 - Accuracy: 0.9169`


## Convolution-Network

```kotlin
val cnNetwork = Network(ImageStringInputLayer(), listOf(
        Conv2DLayer(Shape(intArrayOf(28, 28, 1)), Shape(intArrayOf(27, 27, 4)), Shape(intArrayOf(2, 2, 1)), 4),
        FlatternLayer(Shape(intArrayOf(1, 2916))),
        FullyConnectedLayer(Shape(intArrayOf(1, 2916)), Shape(intArrayOf(1, 10))),
        SoftmaxLayer(Shape(intArrayOf(1, 10)))
))
```
mit folgenden Hyperparametern:

- Batchsize: 256
- Learning-Rate: 0,001
- Epochen: 5
- Loss: Cross-Entropy
- Optimizer: Stochastic Gradient Descent

Ergebnis: `Validation finished - Loss: 1.6226407 - Accuracy: 0.891`
