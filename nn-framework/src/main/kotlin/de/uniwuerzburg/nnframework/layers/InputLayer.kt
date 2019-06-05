package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

class InputLayer<T : Collection<Float>>(private val inputShape: Shape) {
    fun forward(rawDataList: List<T>): List<Tensor>? {
        return rawDataList.map { Tensor(inputShape, it.toFloatArray()) }
    }
}