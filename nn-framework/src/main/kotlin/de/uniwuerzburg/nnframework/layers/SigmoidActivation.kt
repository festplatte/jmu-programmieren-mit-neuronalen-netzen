package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * @author se
 */

/**
 * Repräsentiert eine Sigmoid Aktivierungsfunktion.
 */
class SigmoidActivation(override val outputShape: Shape): ActivationLayer {
    /**
     * Wendet Aktivierungsfunktion an.
     */
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        for (i in inTensors.indices){
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            for (k in inTensor.elements.indices){
                outTensor.elements[k] = sigmoid(inTensor.elements[k])
            }
        }
    }

    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        for (i in inTensors.indices){
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            for (k in inTensor.deltas.indices){
                inTensor.deltas[k] = sigmoid(inTensor.elements[k]) *
                        (1f - sigmoid(inTensor.elements[k])) *
                        outTensor.deltas[k]
            }
        }
    }

    fun sigmoid(value: Float): Float{
        return 1f / (1f + Math.pow(Math.E, (-1f)*value.toDouble()).toFloat())
    }
}