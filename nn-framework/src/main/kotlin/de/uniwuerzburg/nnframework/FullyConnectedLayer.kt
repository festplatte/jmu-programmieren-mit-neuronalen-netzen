package de.uniwuerzburg.nnframework

class FullyConnectedLayer(private val weightmatrix: Tensor,
                          private val bias: Tensor,
                          private val inShape: Shape,
                          private val outShape: Shape) : Layer {


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