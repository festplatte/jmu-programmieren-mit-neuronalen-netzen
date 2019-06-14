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

    /**
     * Berechnet die Ableitung der Loss-Funktion nach den labels.
     */
    fun differentiate(results: List<Tensor>, labels: List<Tensor>)
}