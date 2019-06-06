package de.uniwuerzburg.nnframework.data

/**
 * Beschreibt die Form eines Tensors.
 */
class Shape(private val axis: IntArray) {
    val volume: Int get() {
        var result = 1
        axis.forEach { result *= it }
        return result
    }
    val dimensions: Int get() = axis.size

    fun get(i: Int) = if (i < axis.size ) axis[i] else 1
}