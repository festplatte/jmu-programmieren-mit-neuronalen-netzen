package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Interface, dass implementiert werden sollte, um eine Aktivierungsfunktion (z.B. ReLU, Sigmoid, TanH) zu erstellen.
 */
abstract class ActivationLayer(override val outputShape: Shape): Layer {
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        for (i in inTensors.indices) {
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            for (k in inTensor.elements.indices) {
                outTensor.elements[k] = calculate(inTensor.elements[k])
            }
        }
    }

    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        for (i in inTensors.indices) {
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            for (k in inTensor.deltas.indices) {
                inTensor.deltas[k] = differentiate(inTensor.elements[k]) * outTensor.deltas[k]
            }
        }
    }

    internal abstract fun calculate(value: Float): Float
    internal abstract fun differentiate(value: Float): Float
}