package de.uniwuerzburg.nnframework

class InputLayer<T>(private val inputShape: Shape) {
    fun forward(rawDataList: List<T>): List<Tensor>? {
        // TODO
        // return rawDataList.map { Tensor(inputShape, it) }
        return null
    }
}