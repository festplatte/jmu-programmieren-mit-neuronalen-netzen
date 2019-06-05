package de.uniwuerzburg.nnframework.loss

import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Interface für Loss-Funktionen.
 */
interface LossFkt {
    /**
     * Berechnet die Loss-Funktion
     */
    fun calculate(results: List<Tensor>, labels: List<Tensor>): Float
}