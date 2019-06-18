package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Interface for Layers that use weights and have to update them.
 */
interface WeightLayer: Layer {
    fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>)
}