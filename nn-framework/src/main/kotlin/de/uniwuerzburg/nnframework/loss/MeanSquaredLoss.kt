package de.uniwuerzburg.nnframework.loss

import de.uniwuerzburg.nnframework.data.Tensor
import kotlin.math.pow

class MeanSquaredLoss: LossFkt {
    /**
     * Berechnet die Loss-Funktion
     */
    override fun calculate(results: List<Tensor>, labels: List<Tensor>): Float {
        var combinedLoss = 0f

        for(i in results.indices){
            var result = results.get(i)
            var label = labels.get(i)

            for(k in result.elements.indices){
                combinedLoss += 0.5f * (result.elements[k] - label.elements[k]).pow(2f)
            }
        }

        return combinedLoss / results.size
    }

    /**
     * Berechnet die Ableitung der Loss-Funktion nach den labels.
     */
    override fun differentiate(results: List<Tensor>, labels: List<Tensor>) {
        for(i in results.indices){
            var result = results.get(i)
            var label = labels.get(i)

            for(k in result.elements.indices){
                result.deltas[k] = label.elements[k] - result.elements[k]
            }
        }
    }
}