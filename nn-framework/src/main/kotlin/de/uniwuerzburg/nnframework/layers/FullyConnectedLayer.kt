package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * zu klären:
 * - warum inShape und outShape? die Tensoren haben bereits eine Shape.
 */
class FullyConnectedLayer(private val weightmatrix: Tensor,
                          private val bias: Tensor,
                          private val inShape: Shape,
                          private val outShape: Shape) : Layer {


    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        // TODO inTensors * weightmatrix + bias = outTensors
    }

    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        // TODO berechne die deltas der inTensors mit den deltas der outTensors und Daten beider Tensoren
    }

    override fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        // TODO berechne die Deltas der Gewichte mit Hilfe der Deltas der outTensors und Daten der inTensors
    }
}