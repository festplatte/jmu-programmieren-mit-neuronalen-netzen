package de.uniwuerzburg.nnframework.data

// TODO Zugriff auf axis bauen
class Shape(private val axis: IntArray) {
    val volume: Int get() {
        var result = 1
        axis.forEach { result *= it }
        return result
    }
}