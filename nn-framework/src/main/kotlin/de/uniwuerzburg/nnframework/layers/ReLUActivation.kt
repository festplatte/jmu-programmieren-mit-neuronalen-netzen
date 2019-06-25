package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Aktivierungsfunktion ReLU.
 */
class ReLUActivation(override val outputShape: Shape): ActivationLayer {
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        for (i in inTensors.indices) {
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            for (k in inTensor.elements.indices) {
                outTensor.elements[k] = relu(inTensor.elements[k])
            }
        }
    }

    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        for (i in inTensors.indices) {
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            for (k in inTensor.deltas.indices) {
                inTensor.deltas[k] = dRelu(inTensor.elements[k]) * outTensor.deltas[k]
            }
        }
    }

    private fun relu(value: Float): Float {
        return if (value > 0f) value else 0f
    }

    private fun dRelu(value: Float): Float {
        return if (value > 0f) 1f else 0f
    }
}