package de.uniwuerzburg.nnframework.loss

import de.uniwuerzburg.nnframework.data.Tensor

/**
 * @author se
 */

class CrossEntropyLoss: LossFkt {
    /**
     * Berechnet die Loss-Funktion
     */
    override fun calculate(results: List<Tensor>, labels: List<Tensor>): Float {
        var combinedLoss = 0f

        for(i in results.indices){
            var result = results.get(i)
            var label = labels.get(i)
            combinedLoss += Math.log(result.elements[label.elements[0].toInt()].toDouble()).toFloat()
        }

        return combinedLoss / results.size
    }

    /**
     * Berechnet die Ableitung der Loss-Funktion nach den labels.
     */
    override fun differentiate(results: List<Tensor>, labels: List<Tensor>) {
        // TODO
    }
}