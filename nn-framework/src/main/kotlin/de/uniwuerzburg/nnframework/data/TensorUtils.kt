package de.uniwuerzburg.nnframework.data

import java.util.*
import de.uniwuerzburg.nnframework.mapInPlace

/*
* vb: in work -> add mult with transposed
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
    if (!tensorA.shape.axis.contentEquals(tensorB.shape.axis) || !tensorA.shape.axis.contentEquals(outTensor.shape.axis)) {
        throw IllegalArgumentException("shapes of all tensors need to be equal")
    }

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

fun multAndTransposeFirst(tensorA: Tensor, tensorB: Tensor,
                           tensorA_useDeltas: Boolean = false, tensorB_useDeltas: Boolean = false): Tensor {
    val dimensions = mutableListOf(tensorA.shape.get(1))
    if (tensorB.shape.dimensions > 1) dimensions.add(tensorB.shape.get(1))
    // TODO kopiere weitere Dimensionen für 2+ dimensionale Tensoren

    val resultShape = Shape(dimensions.toIntArray())
    val resultTensor = Tensor(resultShape, FloatArray(resultShape.volume))
    multAndTransposeFirst(tensorA, tensorB, resultTensor, tensorA_useDeltas, tensorB_useDeltas, false)

    return resultTensor
}

fun multAndTransposeSecond(tensorA: Tensor, tensorB: Tensor,
                           tensorA_useDeltas: Boolean = false, tensorB_useDeltas: Boolean = false): Tensor {
    val dimensions = mutableListOf(tensorA.shape.get(0))
    if (tensorB.shape.dimensions > 1) dimensions.add(tensorB.shape.get(0))
    // TODO kopiere weitere Dimensionen für 2+ dimensionale Tensoren

    val resultShape = Shape(dimensions.toIntArray())
    val resultTensor = Tensor(resultShape, FloatArray(resultShape.volume))
    multAndTransposeSecond(tensorA, tensorB, resultTensor, tensorA_useDeltas, tensorB_useDeltas, false)

    return resultTensor
}


/**
 * Multipliziert zwei Tensoren. Dabei wird Matrixmultiplikation in den ersten beiden Dimensionen angewendet.
 * Über evtl. weitere Dimensionen wird nur iteriert.
 */
fun mult(tensorA: Tensor, tensorB: Tensor, outTensor: Tensor) {
    if (tensorA.shape.get(1) != tensorB.shape.get(0)) {
        throw IllegalArgumentException("colums of tensorA must be equal to rows of tensorB")
    }

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

/**
 * Multipliziert zwei Tensoren, wobei der erste Tensor dabei transponiert wird.
 * Es wird Matrixmultiplikation in den ersten beiden Dimensionen angewendet.
 * Über evtl. weitere Dimensionen wird nur iteriert.
 * Falls useDeltas auf true gesetzt wird, wird anstelle des element-Arrays mit dem delta-Array gearbeitet
 * Falls outTensor_sumUp auf true gesetzt wird, wird das Ergebnis zum exisitierenden Array (element oder deltas)
 * des outTensor dazu addiert statt das bisherige Feld einfach zu ersetzen
 */
fun multAndTransposeFirst(tensorA: Tensor, tensorB: Tensor, outTensor: Tensor,
                           tensorA_useDeltas: Boolean = false, tensorB_useDeltas: Boolean = false,
                           outTensor_useDeltas: Boolean =false, outTensor_sumUp: Boolean = false) {

    if (tensorA.shape.get(0) != tensorB.shape.get(0)) {
        throw IllegalArgumentException("rows of tensorA must be equal the rows of tensorB (as it is transposed)")
    }

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
                for (i in 0..tensorA.shape.get(0)-1) { //get(0) because of transposition!
                    //Switch the indices for tensor A
                    if(tensorA_useDeltas && tensorB_useDeltas){
                        println("Attention: This case should actually never occur")
                        result += tensorA.deltas[tensorA.calcIndex(intArrayOf(i, row)) + offsettensorA] * tensorB.deltas[tensorB.calcIndex(intArrayOf(i, column)) + offsetMult]
                    }else if(tensorA_useDeltas){
                        result += tensorA.deltas[tensorA.calcIndex(intArrayOf(i, row)) + offsettensorA] * tensorB.elements[tensorB.calcIndex(intArrayOf(i, column)) + offsetMult]
                    }else if(tensorB_useDeltas){
                        result += tensorA.elements[tensorA.calcIndex(intArrayOf(i, row)) + offsettensorA] * tensorB.deltas[tensorB.calcIndex(intArrayOf(i, column)) + offsetMult]
                    }else{
                        // No deltas involved
                        result += tensorA.elements[tensorA.calcIndex(intArrayOf(i, row)) + offsettensorA] * tensorB.elements[tensorB.calcIndex(intArrayOf(i, column)) + offsetMult]
                    }

                }
                if(!outTensor_useDeltas){
                    if(!outTensor_sumUp){
                        outTensor.elements[outTensor.calcIndex(intArrayOf(row, column)) + offsetOut] = result
                    }else{
                        outTensor.elements[outTensor.calcIndex(intArrayOf(row, column)) + offsetOut] += result
                    }

                }else{
                    if(!outTensor_sumUp){
                        outTensor.deltas[outTensor.calcIndex(intArrayOf(row, column)) + offsetOut] = result
                    }else{
                        outTensor.deltas[outTensor.calcIndex(intArrayOf(row, column)) + offsetOut] += result
                    }

                }
            }
        }

        offsettensorA += voltensorAMatrix
        offsetMult += volMultMatrix
        offsetOut += volOutMatrix
    }
}



/**
 * Multipliziert zwei Tensoren, wobei der zweite Tensor dabei transponiert wird.
 * Es wird Matrixmultiplikation in den ersten beiden Dimensionen angewendet.
 * Über evtl. weitere Dimensionen wird nur iteriert.
 * Falls useDeltas auf true gesetzt wird, wird anstelle des element-Arrays mit dem delta-Array gearbeitet
 */
fun multAndTransposeSecond(tensorA: Tensor, tensorB: Tensor, outTensor: Tensor,
                           tensorA_useDeltas: Boolean = false, tensorB_useDeltas: Boolean = false,
                           outTensor_useDeltas: Boolean =false) {
    if (tensorA.shape.get(1) != tensorB.shape.get(1)) {
        throw IllegalArgumentException("colums of tensorA must be equal the columns of tensorB (as it is transposed)")
    }

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
                    //Switch the indices for tensor B
                    if(tensorA_useDeltas && tensorB_useDeltas){
                        println("Attention: This case should actually never occur")
                        result += tensorA.deltas[tensorA.calcIndex(intArrayOf(row, i)) + offsettensorA] * tensorB.deltas[tensorB.calcIndex(intArrayOf(column, i)) + offsetMult]
                    }else if(tensorA_useDeltas){
                        result += tensorA.deltas[tensorA.calcIndex(intArrayOf(row, i)) + offsettensorA] * tensorB.elements[tensorB.calcIndex(intArrayOf(column, i)) + offsetMult]
                    }else if(tensorB_useDeltas){
                        result += tensorA.elements[tensorA.calcIndex(intArrayOf(row, i)) + offsettensorA] * tensorB.deltas[tensorB.calcIndex(intArrayOf(column, i)) + offsetMult]
                    }else{
                        // No deltas involved
                        result += tensorA.elements[tensorA.calcIndex(intArrayOf(row, i)) + offsettensorA] * tensorB.elements[tensorB.calcIndex(intArrayOf(column, i)) + offsetMult]
                    }

                }
                if(!outTensor_useDeltas){
                    outTensor.elements[outTensor.calcIndex(intArrayOf(row, column)) + offsetOut] = result
                }else{
                    outTensor.deltas[outTensor.calcIndex(intArrayOf(row, column)) + offsetOut] = result
                }
            }
        }

        offsettensorA += voltensorAMatrix
        offsetMult += volMultMatrix
        offsetOut += volOutMatrix
    }
}

/**
 * Initialisiert die Elemente des uebergebenen Tensors mit gleichverteilt, zufaelligen Werten zwischen -1 und 1
 */
fun initializeWeights(tensor:Tensor){
    tensor.elements.mapInPlace {_ ->
        //ThreadLocalRandom.current().nextFloat()}
        if(Random().nextDouble()<0.5){
            Random().nextFloat()
        }else{
            Random().nextFloat() * -1f
        }
    }
}

/**
 *  Gibt die Inhalte des Tensors aus, solange er nur maximal 2 Dimensionen hat
 */
fun printTensor(tensor: Tensor){
    if(tensor.shape.dimensions == 1){
        var output = "("
        for (element in tensor.elements.iterator()){
            output = output + element + ", "
        }
        output = output.subSequence(0, output.length-2).toString() + ")^T"
        println(output)
    }
    else if(tensor.shape.dimensions == 2)
    {
        for (x in 0 until tensor.shape.get(0)){
            for (y in 0 until tensor.shape.get(1)){
                print(tensor.get(x,y))
                print("\t\t")
            }
            println()
        }
    }else{
        //More than two dimensions
        //TODO fill if needed
        println("Too many dimenstions for pretty print")
    }

}


