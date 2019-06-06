package de.uniwuerzburg.nnframework

import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.layers.InputLayer
import de.uniwuerzburg.nnframework.layers.Layer

/**
 * offene Fragen:
 * - was machen die deltaParams?
 */
class Network<T: Collection<Float>>(private val input: InputLayer<T>,
                                    private val layers: List<Layer>,
                                    private val deltaParams: List<Tensor>) {

    val parameters get() = deltaParams
    
    // TODO implement caches if required

    fun forward() {
        // TODO rufe forward der einzelnen Layer nacheinander auf
    }
    fun backprop(labels: T) {
        // TODO Berechnet Loss, backward jedes Layers, calculateDeltaWeights der Layer
    }

}