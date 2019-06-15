package de.uniwuerzburg.nnframework.data

/*
* vb: in work -> add initWeights functionality
* */

fun add(tensorA: Tensor, tensorB: Tensor): Tensor {
    val resultShape = Shape(tensorA.shape.axis.clone())
    val resultTensor = Tensor(resultShape, FloatArray(resultShape.volume))
    add(tensorA, tensorB, resultTensor)

    return resultTensor
}

/**
 * Addiert Vektoren, d.h. nur die erste Dimension der Tensoren wird addiert.
 */
fun add(tensorA: Tensor, tensorB: Tensor, outTensor: Tensor) {
    val volume = outTensor.shape.get(0)
    var offset = 0

    while (offset < outTensor.shape.volume) {
        for (row in 0..tensorA.shape.get(0)-1) {
            outTensor.elements[outTensor.calcIndex(intArrayOf(row)) + offset] = tensorA.elements[tensorA.calcIndex(intArrayOf(row)) + offset] + tensorB.elements[tensorB.calcIndex(intArrayOf(row)) + offset]
        }
        offset += volume
    }
}

fun mult(tensorA: Tensor, tensorB: Tensor): Tensor {
    val dimensions = mutableListOf(tensorA.shape.get(0))
    if (tensorB.shape.dimensions > 1) dimensions.add(tensorB.shape.get(1))
    // TODO kopiere weitere Dimensionen für 2+ dimensionale Tensoren

    val resultShape = Shape(dimensions.toIntArray())
    val resultTensor = Tensor(resultShape, FloatArray(resultShape.volume))
    mult(tensorA, tensorB, resultTensor)

    return resultTensor
}

/**
 * Multipliziert zwei Tensoren. Dabei wird Matrixmultiplikation in den ersten beiden Dimensionen angewendet.
 * Über evtl. weitere Dimensionen wird nur iteriert.
 */
fun mult(tensorA: Tensor, tensorB: Tensor, outTensor: Tensor) {
    val voltensorAMatrix = tensorA.shape.get(0) * tensorA.shape.get(1)
    val volMultMatrix = tensorB.shape.get(0) * tensorB.shape.get(1)
    val volOutMatrix = outTensor.shape.get(0) * outTensor.shape.get(1)

    var offsettensorA = 0
    var offsetMult = 0
    var offsetOut = 0

    while (offsetOut < outTensor.shape.volume) {
        var result: Float
        for (row in 0..outTensor.shape.get(0)-1) {
            for (column in 0..outTensor.shape.get(1)-1) {
                result = 0f
                for (i in 0..tensorA.shape.get(1)-1) {
                    result += tensorA.elements[tensorA.calcIndex(intArrayOf(row, i)) + offsettensorA] * tensorB.elements[tensorB.calcIndex(intArrayOf(i, column)) + offsetMult]
                }
                outTensor.elements[outTensor.calcIndex(intArrayOf(row, column)) + offsetOut] = result
            }
        }

        offsettensorA += voltensorAMatrix
        offsetMult += volMultMatrix
        offsetOut += volOutMatrix
    }
}
