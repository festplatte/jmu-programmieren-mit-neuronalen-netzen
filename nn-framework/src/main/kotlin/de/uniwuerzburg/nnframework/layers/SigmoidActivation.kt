package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import kotlin.math.E
import kotlin.math.pow

/**
 * @author se
 */

/**
 * Repräsentiert eine Sigmoid Aktivierungsfunktion.
 */
class SigmoidActivation(outputShape: Shape) : ActivationLayer(outputShape) {
    override fun calculate(value: Float): Float {
        return 1f / (1f + E.toFloat().pow((-1f) * value))
    }

    override fun differentiate(value: Float): Float {
        val sigmoid = calculate(value)
        return sigmoid * (1f - sigmoid)
    }
}