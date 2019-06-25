package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import kotlin.math.pow
import kotlin.math.tanh

/**
 * Aktivierungsfunktion tanh.
 */
class TanHActivation(override val outputShape: Shape): ActivationLayer {
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        for (i in inTensors.indices) {
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            for (k in inTensor.elements.indices) {
                outTensor.elements[k] = tanh(inTensor.elements[k])
            }
        }
    }

    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        for (i in inTensors.indices) {
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            for (k in inTensor.deltas.indices) {
                inTensor.deltas[k] = (1 - tanh(inTensor.elements[k]).pow(2)) * outTensor.deltas[k]
            }
        }
    }
}