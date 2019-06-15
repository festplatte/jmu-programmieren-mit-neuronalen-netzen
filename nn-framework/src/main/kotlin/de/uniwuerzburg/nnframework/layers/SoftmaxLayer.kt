package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * @author se
 */

class SoftmaxLayer(override val outputShape: Shape): ActivationLayer {

    /**
     * Wendet die Softmax-Funktion elementweise an
     */
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        for (i in inTensors.indices){
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            var softmaxsum = 0f
            for (element in inTensor.elements){
                softmaxsum += Math.pow(Math.E, element.toDouble()).toFloat()
            }
            for (k in outTensor.elements.indices){
                outTensor.elements[k] = Math.pow(Math.E, inTensor.elements[k].toDouble()).toFloat() / softmaxsum
            }
        }
    }

    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        for (i in inTensors.indices){
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)

            for (k in inTensor.deltas.indices){
                var inDelta = 0f
                for(j in outTensor.deltas.indices){
                    //delta e_j * (delta e_k / delta d_j)
                    if (k == j){
                        inDelta += outTensor.deltas[j] * outTensor.elements[k] * (1f - outTensor.elements[j])
                    }
                    else{
                        inDelta += outTensor.deltas[j] * outTensor.elements[k] * (-1f) * outTensor.elements[j]
                    }
                }
                inTensor.deltas[k] = inDelta
            }
        }
    }
}