package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

interface WeightLayer: Layer {
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>)
    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>)
    fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>)
}