package de.uniwuerzburg.nnframework.loss

import de.uniwuerzburg.nnframework.data.Tensor
import kotlin.math.ln

/**
 * @author se
 */

class CrossEntropyLoss : LossFkt {
    /**
     * Berechnet die Loss-Funktion
     */
    override fun calculate(results: List<Tensor>, labels: List<Tensor>): Float {
        var combinedLoss = 0f

        for (i in results.indices) {
            var result = results.get(i)
            var label = labels.get(i)

            for (k in result.elements.indices) {
                combinedLoss -= label.elements[k] * ln(result.elements[k])
            }
        }

        return combinedLoss / results.size
    }

    /**
     * Berechnet die Ableitung der Loss-Funktion nach den labels.
     */
    override fun differentiate(results: List<Tensor>, labels: List<Tensor>) {
        for (i in results.indices) {
            var result = results.get(i)
            var label = labels.get(i)

            for (k in result.elements.indices) {
                result.deltas[k] = (-1) * (label.elements[k] / result.elements[k])
            }
        }
    }
}