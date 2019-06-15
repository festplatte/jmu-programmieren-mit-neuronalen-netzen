package de.uniwuerzburg.nnframework.data

import java.util.*
import de.uniwuerzburg.nnframework.mapInPlace

/*
* vb: in work -> add initWeights and print functionalities
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

/**
 * Initialisiert die Elemente des uebergebenen Tensors mit gleichverteilt, zufaelligen Werten zwischen -1 und 1
 */
fun initializeWeights(tensor:Tensor){
    tensor.elements.mapInPlace {Float ->
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
        output = output.subSequence(0, output.length-2).toString() + ")"
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


