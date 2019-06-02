package de.uniwuerzburg.nnframework

class Shape(private val axis: IntArray) {
    val volume: Int get() {
        var result = 1
        axis.forEach { result *= it }
        return result
    }
}