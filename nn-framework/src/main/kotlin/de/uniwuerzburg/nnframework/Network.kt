package de.uniwuerzburg.nnframework

class Network<T>(private val input: InputLayer<T>,
                 private val layers: List<Layer>,
                 private val deltaParams: List<Tensor>) {

    val parameters get() = deltaParams

    // TODO implement caches if required

    fun forward() {
        // TODO
    }
    fun backprop(labels: T) {
        // TODO
    }

}