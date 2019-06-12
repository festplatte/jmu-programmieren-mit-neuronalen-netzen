package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

class SoftmaxLayer(override val outputShape: Shape): Layer {
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        // TODO
    }

    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        // TODO
    }
    override fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        // TODO
    }
}