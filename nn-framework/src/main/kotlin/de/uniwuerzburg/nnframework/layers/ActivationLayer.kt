package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Repräsentiert eine Aktivierungsfunktion (z.B. ReLU, Sigmoid, TanH). Sollte evtl. als Interface
 * implementiert werden.
 */
class ActivationLayer: Layer {
    /**
     * Wendet Aktivierungsfunktion an.
     */
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