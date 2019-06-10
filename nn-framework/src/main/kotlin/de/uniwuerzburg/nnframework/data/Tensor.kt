package de.uniwuerzburg.nnframework.data

/**
 * Ein Tensor ist ein Container für Float Werte, die entsprechend der Shape angeordnet sind.
 * Somit können Vektoren, Matrizen und mehrdimensionale Datenformen abgebildet werden.
 *
 * @author mg
 */
class Tensor(internal var shape: Shape, private var elements: FloatArray) {
    private val deltas: FloatArray by lazy { FloatArray(shape.volume) }

    // TODO evtl. die Rechenoperationen aus der Tensorklasse in die Layer oder Utils auslagern -> Rechenoperationen sind nicht allegemein, z.b. Matrixmultiplikation in ersten beiden Dimensionen

    fun add(tensor: Tensor): Tensor {
        val resultShape = Shape(this.shape.axis.clone())
        val resultTensor = Tensor(resultShape, FloatArray(resultShape.volume))
        add(tensor, resultTensor)

        return resultTensor
    }

    /**
     * Addiert Vektoren, d.h. nur die erste Dimension der Tensoren wird addiert.
     */
    fun add(addTensor: Tensor, outTensor: Tensor) {
        val volume = outTensor.shape.get(0)
        var offset = 0

        while (offset < outTensor.shape.volume) {
            for (row in 0..this.shape.get(0)-1) {
                outTensor.elements[outTensor.calcIndex(intArrayOf(row)) + offset] = this.elements[this.calcIndex(intArrayOf(row)) + offset] + addTensor.elements[addTensor.calcIndex(intArrayOf(row)) + offset]
            }
            offset += volume
        }
    }

    fun mult(tensor: Tensor): Tensor {
        val dimensions = mutableListOf(this.shape.get(0))
        if (tensor.shape.dimensions > 1) dimensions.add(tensor.shape.get(1))
        // TODO kopiere weitere Dimensionen für 2+ dimensionale Tensoren

        val resultShape = Shape(dimensions.toIntArray())
        val resultTensor = Tensor(resultShape, FloatArray(resultShape.volume))
        mult(tensor, resultTensor)

        return resultTensor
    }

    /**
     * Multipliziert zwei Tensoren. Dabei wird Matrixmultiplikation in den ersten beiden Dimensionen angewendet.
     * Über evtl. weitere Dimensionen wird nur iteriert.
     */
    fun mult(multTensor: Tensor, outTensor: Tensor) {
        // vermutlich nicht performant wenn jedes mal gecheckt wird
//        if (this.shape.dimensions <= 2
//                || this.shape.get(1) != tensor.shape.get(0)
//                || this.shape.dimensions != tensor.shape.dimensions) {
//            throw IllegalArgumentException()
//        }
        val volThisMatrix = this.shape.get(0) * this.shape.get(1)
        val volMultMatrix = multTensor.shape.get(0) * multTensor.shape.get(1)
        val volOutMatrix = outTensor.shape.get(0) * outTensor.shape.get(1)

        var offsetThis = 0
        var offsetMult = 0
        var offsetOut = 0

        while (offsetOut < outTensor.shape.volume) {
            var result: Float
            for (row in 0..outTensor.shape.get(0)-1) {
                for (column in 0..outTensor.shape.get(1)-1) {
                    result = 0f
                    for (i in 0..this.shape.get(1)-1) {
                        result += this.elements[this.calcIndex(intArrayOf(row, i)) + offsetThis] * multTensor.elements[multTensor.calcIndex(intArrayOf(i, column)) + offsetMult]
                    }
                    outTensor.elements[outTensor.calcIndex(intArrayOf(row, column)) + offsetOut] = result
                }
            }

            offsetThis += volThisMatrix
            offsetMult += volMultMatrix
            offsetOut += volOutMatrix
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
    private fun calcIndex(indices: IntArray): Int {
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
}