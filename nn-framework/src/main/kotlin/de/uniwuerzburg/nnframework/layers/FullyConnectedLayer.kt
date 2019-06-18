package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.*

/**
 * @author vb
 */

/**
 * Die Tensoren haben zwar eine Shape, inShape und outShape sind aber zur Initialisierung der Weightmatrix
 */
class FullyConnectedLayer(private val inShape: Shape,
                          private val outShape: Shape) : WeightLayer {

    // Create tensors for the weight matrix W and for the bias
    private var bias: Tensor = Tensor(Shape(outShape.axis.clone()), FloatArray(outShape.volume))

    // W: Fully connected, e.g. one weight between each pair contained in the in and the outShape
    // For each element in the inTensor there is a connection to each element of the outTensor
    // If the inshape has more than one dimensions, the shape needs to be flattened to a vector
    private var weightmatrix_shape = Shape(intArrayOf(inShape.volume, outShape.volume))
    private var weightmatrix: Tensor = Tensor(weightmatrix_shape)


    init {
        // Initialize the weights
        initializeWeights(bias)
        initializeWeights(weightmatrix)
    }

    override val outputShape get() = outShape
    val getBias get() = bias
    val getWeights get() = weightmatrix


    /*
    * The forward pass fills the elements of the outTensors,
    * while having access to the elements in inTensors:
    * inTensors * weightmatrix + bias = outTensors
    */
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        for (i in inTensors.indices){
            val inTensor = inTensors.get(i)
            val outTensor = outTensors.get(i)
            mult(inTensor, weightmatrix, outTensor)
            add(outTensor,bias, outTensor)
        }

        /*
        val inTensorsIterator = inTensors.iterator()
        for (inTensor in inTensorsIterator) {
            //Returns a tensor, however the existing outTensor should be filled ...
            mult(inTensor, weightmatrix) //.add(bias)
        }*/

    }

    /*
    * The backward pass fills in the deltas of the inTensors,
    * while having access to the deltas in outTensors as well as to the elements of the inTensors
    * deltaX = deltaY * W^T
    */
    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        for (i in inTensors.indices){
            val inTensor = inTensors.get(i)
            val outTensor = outTensors.get(i)
            //use deltas of the outTensor and write the result to the deltas of the inTensor
            multAndTransposeSecond(outTensor,weightmatrix,inTensor,
                    tensorA_useDeltas = true,tensorB_useDeltas = false, outTensor_useDeltas = true)
        }
    }

    /*
    * The calculateDeltaWeights function calculates the delta weights by
    * using the elements of the inTensors and the deltas of the outTensors
    * DeltaBias = DeltaY
    * DeltaW = X^T * DeltaY
    *
    * If the list contains more than one element, the update values are summed up and averaged at the end
    */
    override fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        val biasDeltas = FloatArray(bias.shape.volume)
        for (i in inTensors.indices){
            val inTensor = inTensors.get(i)
            val outTensor = outTensors.get(i)

            // Add the delta weights for the bias and add them
            val currentBiasDeltas = outTensor.deltas
            for (j in 0 until currentBiasDeltas.size){
                biasDeltas[j] += currentBiasDeltas[j]
            }

            //Calculate the delta weight for W and add them
            multAndTransposeFirst(inTensor, outTensor, weightmatrix,
                    tensorA_useDeltas = false, tensorB_useDeltas = true, outTensor_useDeltas = true,
                    outTensor_sumUp = true)
        }

        // The weight deltas are already filled but the bias deltas still net to be written
        bias.setDeltas(biasDeltas)
    }

    fun setWeightsForTesting(bias:Tensor, weights:Tensor){
        this.bias = bias
        this.weightmatrix = weights
    }
}