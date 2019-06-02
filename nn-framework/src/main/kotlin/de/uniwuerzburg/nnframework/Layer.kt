package de.uniwuerzburg.nnframework

interface Layer {
    fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>)
    fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>)
    fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>)
}