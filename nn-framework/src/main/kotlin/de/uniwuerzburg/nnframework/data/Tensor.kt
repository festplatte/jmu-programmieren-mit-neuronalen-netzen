package de.uniwuerzburg.nnframework.data

class Tensor(private var shape: Shape, private var elements: FloatArray) {
    private val deltas: FloatArray by lazy { FloatArray(shape.volume) }

    // TODO evtl. Zugriffsmethode auf Daten in Achsenschreibweise (x, y, z, ...)
    // TODO Multiplikation von Tensoren implementieren
    // TODO Addition von Tensoren (für Bias) implementieren
}