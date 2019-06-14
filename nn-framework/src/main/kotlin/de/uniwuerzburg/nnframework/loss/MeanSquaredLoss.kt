package de.uniwuerzburg.nnframework.loss

import de.uniwuerzburg.nnframework.data.Tensor

class MeanSquaredLoss: LossFkt {
    /**
     * Berechnet die Loss-Funktion
     */
    override fun calculate(results: List<Tensor>, labels: List<Tensor>): Float {
        // TODO
        return 0f
    }

    /**
     * Berechnet die Ableitung der Loss-Funktion nach den labels.
     */
    override fun differentiate(results: List<Tensor>, labels: List<Tensor>) {
        // TODO
    }
}