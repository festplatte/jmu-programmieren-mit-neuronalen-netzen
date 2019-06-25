package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import kotlin.math.pow
import kotlin.math.tanh

/**
 * Aktivierungsfunktion tanh.
 */
class TanHActivation(outputShape: Shape): ActivationLayer(outputShape) {
    override fun calculate(value: Float): Float {
        return tanh(value)
    }

    override fun differentiate(value: Float): Float {
        return 1f - tanh(value).pow(2)
    }
}