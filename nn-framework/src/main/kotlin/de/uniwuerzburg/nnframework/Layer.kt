package de.uniwuerzburg.nnframework

interface Layer {
    fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>)

    /**
     * Berechnung der Fehler für die aktuelle Schicht.
     * @param inTensors beinhaltet Deltas der vorherigen Schicht (keine Ahnung,
     * ob die Elemente die Daten oder Gewichte der vorherigen Schicht sind.)
     * @param outTensors Fehler/Deltas der aktuellen Schicht (wieder entweder mit
     * Daten oder Gewchten)
     */
    fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>)
    fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>)
}