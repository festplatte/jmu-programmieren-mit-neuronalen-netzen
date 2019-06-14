package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

interface Layer {
    val outputShape: Shape

    fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>)

    /**
     * Berechnung der Deltas für die aktuelle Schicht.
     * @param outTensors beinhalten bereits die Deltas der nachfolgenden Schicht des Forward-Passes.
     * @param inTensors Deltas der inTensoren werden berechnet und gespeichert
     */
    fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>)
}