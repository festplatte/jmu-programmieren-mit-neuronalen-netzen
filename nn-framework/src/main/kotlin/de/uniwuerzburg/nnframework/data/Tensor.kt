package de.uniwuerzburg.nnframework.data

/**
 * Ein Tensor ist ein Container für Float Werte, die entsprechend der Shape angeordnet sind.
 * Somit können Vektoren, Matrizen und mehrdimensionale Datenformen abgebildet werden.
 */
class Tensor(val shape: Shape, var elements: FloatArray = FloatArray(shape.volume)) {
    private var _deltas: FloatArray? = null
    var deltas: FloatArray
        get() {
            if (_deltas == null) {
                _deltas = FloatArray(shape.volume)
            }
            return _deltas ?: throw Exception("someone screwed up the deltas!")
        }
        set(value) {
            _deltas = value
        }


    init {
        if (elements.size != shape.volume) {
            throw IllegalArgumentException("elements size must match the volume of shape")
        }
    }

    /**
     * Berechnet den Index an der angegebenen Position. Für jede Dimension bzw. Achse des Tensors muss ein Zugriffswert geliefert werden.
     *
     * Beispiel: 4x3x2 Matrix
     * (0  4  8  | 12 16 20 )
     * (1  5  9  | 13 17 21 )
     * (2  6  10 | 14 18 22 )
     * (3  7  11 | 15 19 23 )
     *
     * Um den Index für das Element 18 zu erhalten, muss `tensor.calcIndex(2, 1, 1)` aufgerufen werden.
     */
    fun calcIndex(indices: IntArray): Int {
        var index = 0
        for (i in indices.indices) {
            var curIndex = indices[i]
            if (i != 0) {
                for (j in 0..i - 1) {
                    curIndex *= shape.get(j)
                }
            }
            index += curIndex
        }
        return index
    }

    /**
     * Gibt das Element des Tensors an der gegebenen Position zurück.
     * @param indices der Zugriffs-Index für jede Dimension/Achse
     */
    fun get(vararg indices: Int): Float {
        return elements[calcIndex(indices)]
    }

    /**
     * Gibt das Delta des Tensors an der gegebenen Position zurück.
     * @param indices der Zugriffs-Index für jede Dimension/Achse
     */
    fun getDelta(vararg indices: Int): Float {
        return deltas[calcIndex(indices)]
    }

    /**
     * Setzt das Element des Tensors an der gegebenen Position.
     * @param value zu setzender Wert
     * @param indices der Zugriffs-Index für jede Dimension/Achse
     */
    fun set(value: Float, vararg indices: Int) {
        elements[calcIndex(indices)] = value
    }

    /**
     * Setzt das Element des Delta-Tensors an der gegebenen Position.
     * @param value zu setzender Wert
     * @param indices der Zugriffs-Index für jede Dimension/Achse
     */
    fun setDelta(value: Float, vararg indices: Int) {
        deltas[calcIndex(indices)] = value
    }
}
