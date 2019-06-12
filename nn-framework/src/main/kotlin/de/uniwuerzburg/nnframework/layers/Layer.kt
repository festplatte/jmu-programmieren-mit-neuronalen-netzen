package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

interface Layer {
    val outputShape: Shape

    fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>)

    /**
     * Berechnung der Fehler für die aktuelle Schicht.
     * @param inTensors beinhaltet Deltas der Daten der vorherigen Schicht
     * @param outTensors Fehler/Deltas der aktuellen Schicht
     */
    fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>)
    fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>)
}