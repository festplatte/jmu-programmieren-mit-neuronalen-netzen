package de.uniwuerzburg.nnframework.data

class Tensor(private var shape: Shape, private var elements: FloatArray) {
    private val deltas: FloatArray by lazy { FloatArray(shape.volume) }

    // TODO Addition von Tensoren (für Bias) implementieren

    fun mult(tensor: Tensor): Tensor {
        val resultShape = Shape(arrayOf(this.shape.get(0), tensor.shape.get(1)).toIntArray())
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

        while (offsetOut > outTensor.shape.volume) {
            var result: Float
            for (row in 0..outTensor.shape.get(0)) {
                for (column in 0..outTensor.shape.get(1)) {
                    result = 0f
                    for (i in 0..this.shape.get(1)) {
                        result += this.elements[this.calcIndex(arrayOf(row, i).toIntArray()) + offsetThis] * multTensor.elements[multTensor.calcIndex(arrayOf(i, column).toIntArray()) + offsetMult]
                    }
                    outTensor.elements[outTensor.calcIndex(arrayOf(row, column).toIntArray()) + offsetOut] = result
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
        if (indices.size != shape.dimensions) {
            throw IndexOutOfBoundsException()
        }
        var index = 0
        for (i in indices.indices) {
            var curIndex = indices[i]
            for (j in i..0) {
                curIndex *= shape.get(j)
            }
            index += curIndex
        }
        return index
    }

    fun get(vararg indices: Int): Float {
        return elements[calcIndex(indices)]
    }

    fun set(value: Float, vararg indices: Int) {
        elements[calcIndex(indices)] = value
    }
}