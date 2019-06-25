package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Aktivierungsfunktion ReLU.
 */
class ReLUActivation(outputShape: Shape): ActivationLayer(outputShape) {
    override fun calculate(value: Float): Float {
        return if (value > 0f) value else 0f
    }

    override fun differentiate(value: Float): Float {
        return if (value > 0f) 1f else 0f
    }
}