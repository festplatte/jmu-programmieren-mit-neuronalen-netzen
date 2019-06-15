package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.*

/**
 * @author vb
 */

/**
 * Die Tensoren haben war eine Shape, inShape und outShape sind aber zur Initialisierung der Weightmatrix
 */
class FullyConnectedLayer(private val inShape: Shape,
                          private val outShape: Shape) : WeightLayer {

    // Create tensors for the weight matrix W and for the bias
    private val bias: Tensor = Tensor(Shape(outShape.axis.clone()), FloatArray(outShape.volume))

    // W: Fully connected, e.g. one weight between each pair contained in the in and the outShape
    // For each element in the inTensor there is a connection to each element of the outTensor
    // If the inshape has more than one dimensions, the shape needs to be flattened to a vector
    private val weightmatrix_shape = Shape(intArrayOf(inShape.volume, outShape.volume))
    private val weightmatrix: Tensor = Tensor(weightmatrix_shape, FloatArray(weightmatrix_shape.volume))


    init {
        // Initialize the weights
        initializeWeights(bias)
        initializeWeights(weightmatrix)
    }

    override val outputShape get() = outShape

    /*
    * The forward pass fills the elements of the outTensors,
    * while having access to the elements in inTensors:
    * inTensors * weightmatrix + bias = outTensors
    */
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        for (i in inTensors.indices){
            var inTensor = inTensors.get(i)
            var outTensor = outTensors.get(i)
            for (k in inTensor.deltas.indices){
                mult(inTensor,weightmatrix,outTensor)
                add(outTensor,bias, outTensor)
            }
        }

        /*
        val inTensorsIterator = inTensors.iterator()
        for (inTensor in inTensorsIterator) {
            //Returns a tensor, however the existing outTensor should be filled ...
            mult(inTensor, weightmatrix) //.add(bias) TODO add bias after Michi has finished his Tensor class
        }*/

    }

    /*
    * The backward pass fills in the deltas of the inTensors,
    * while having access to the deltas in outTensors as well as to the elements of the inTensors
    */
    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        // TODO berechne die deltas der inTensors mit den deltas der outTensors und Daten beider Tensoren
    }

    /*
    * The calculateDeltaWeights function calculates the delta weights by
    * using the elements of the inTensors and the deltas of the outTensors
    */
    override fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        // TODO berechne die Deltas der Gewichte mit Hilfe der Deltas der outTensors und Daten der inTensors
    }
}