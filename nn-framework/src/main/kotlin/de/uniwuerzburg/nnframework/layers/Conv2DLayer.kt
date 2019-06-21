package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.*

/**
 * @author vb
 */

/**
 * At the moment, neither padding nor dilation or strides are implemented
 */
class Conv2DLayer(private val inputShape: Shape,
                  private val outShape: Shape,
                  private val filterShape: Shape,
                  private val numOfFilters: Int) : WeightLayer {

    // TODO add padding, dilation and strides (optionally)

    // Create tensors for the filters and for the bias
    // There is one bias term for each filter
    var bias: Tensor = Tensor(Shape(intArrayOf(1, numOfFilters)))

    // The kernel tensor has one dimension more that the filter Shape, the additional dimension corresponds to the
    // number of filters dimensions.add(tensorB.shape.get(1)
    var kernel: Tensor = Tensor(Shape(intArrayOf(0)))  //Dummy value, will get overwritten during init()

    init {
        // Check validty of parameters
        if(inputShape.dimensions >3){
            throw IllegalArgumentException("The convoltion layer can not deal with input dimensions greater than 3!")
        }

        if (inputShape.get(2)!= filterShape.get(2)) {
            throw IllegalArgumentException("The filter depth must match the input depth!")
        }

        val calculatedOutShape = Shape(intArrayOf(inputShape.get(0) - filterShape.get(0) +1 ,
                                                  inputShape.get(1) - filterShape.get(0) +1 ,
                                                  inputShape.get(2)))

        if(!outShape.equals(calculatedOutShape)){
            throw IllegalArgumentException("The output shape must match the input shape and the filter shape!")
        }

        // The kernel tensor has one dimension more that the filter Shape, the additional dimension corresponds to the
        // number of filters dimensions.add(tensorB.shape.get(1)
        val dimensions = filterShape.axis.clone().toMutableList()
        dimensions.add(numOfFilters)
        val kernel_tensor_shape = Shape(dimensions.toIntArray())
        kernel = Tensor(kernel_tensor_shape)

        // Initialize the weights
        initializeWeights(bias)
        initializeWeights(kernel)
    }



    override val outputShape get() = outShape
    val getBias get() = bias
    val getKernel get() = kernel


    /*
    * The forward pass fills the elements of the outTensors,
    * while having access to the elements in inTensors:
    * outTensor = inTensor * kernelTensor + bias  (*: convolution operator)
    */
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {

        for (i in inTensors.indices){
            val inTensor = inTensors.get(i)
            val outTensor = outTensors.get(i)
            /*
            mult(inTensor, weightmatrix, outTensor)
            add(outTensor,bias, outTensor)

             */
        }

    }

    /*
    * The backward pass fills in the deltas of the inTensors,
    * while having access to the deltas in outTensors as well as to the elements of the inTensors
    * deltaX = deltaY * W^T
    */
    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        /*
        for (i in inTensors.indices){
            val inTensor = inTensors.get(i)
            val outTensor = outTensors.get(i)
            //use deltas of the outTensor and write the result to the deltas of the inTensor
            multAndTransposeSecond(outTensor,weightmatrix,inTensor,
                    tensorA_useDeltas = true,tensorB_useDeltas = false, outTensor_useDeltas = true)
        }*/
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
        /*
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
        */
    }

    fun setWeightsForTesting(bias:Tensor, kernel:Tensor){
        this.bias = bias
        this.kernel = kernel
    }

}