package de.uniwuerzburg.nnframework

class Tensor(private var shape: Shape, private var elements: FloatArray) {
    private val deltas: FloatArray by lazy { FloatArray(shape.volume) }
}