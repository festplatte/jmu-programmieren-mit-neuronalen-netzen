package de.uniwuerzburg.nnframework.data

/**
 * Ein Tensor ist ein Container für Float Werte, die entsprechend der Shape angeordnet sind.
 * Somit können Vektoren, Matrizen und mehrdimensionale Datenformen abgebildet werden.
 */
class Tensor(val shape: Shape, var elements: FloatArray = FloatArray(shape.volume)) {
    val deltas: FloatArray by lazy { FloatArray(shape.volume) }

    init {
        if (elements.size != shape.volume) {
            throw IllegalArgumentException("elements must be of the same size as shape")
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
                for (j in 0..i-1) {
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
     * Setzt das Element des Tensors an der gegebenen Position.
     * @param value zu setzender Wert
     * @param indices der Zugriffs-Index für jede Dimension/Achse
     */
    fun set(value: Float, vararg indices: Int) {
        elements[calcIndex(indices)] = value
    }

    /**
     * Die Funktion kann genutzt werden, um die Deltas des Tensors zu setzen
     */
    fun setDeltas(deltas: FloatArray){
        if(deltas.size != this.deltas.size){
            throw IllegalArgumentException("The delta array size does have the right size")
        }
        for (i in 0 until deltas.size){
            this.deltas[i] = deltas[i]
        }
    }
}
