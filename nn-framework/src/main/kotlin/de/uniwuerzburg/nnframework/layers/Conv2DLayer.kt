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
    var bias: Tensor = Tensor(Shape(intArrayOf(numOfFilters))) //col_vector

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
            convolve(inTensor, kernel, outTensor)
            addBiasPerFilter(outTensor, bias)
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

    /**
     * This function can be used to explicitly set the weights during testing
     */
    fun setWeightsForTesting(bias:Tensor, kernel:Tensor){
        this.bias = bias
        this.kernel = kernel
    }

    /**
     * This function returns the result of applying the convolution operator for images with depth
     * A 2-d convolution ‘convolves’ along two spatial dimensions.
     * It has a really small kernel, essentially a window of pixel values, that slides along those two dimensions.
     * The rgb channel isn't handled as small window of depth, but obtained from beginning to end (first channel to last)
     * That is, even a convolution with a small spatial window of 1x1, which takes a single pixel spatially in the
     * width/height dimensions, would still take all 3 RGB channels
     */
    private fun convolve(inTensor: Tensor, kernel: Tensor, outTensor: Tensor){
        val inputHeight = inTensor.shape.get(0)
        val inputWidth = inTensor.shape.get(1)
        val filterHeight = kernel.shape.get(0)
        val filterWidth = kernel.shape.get(1)
        val imageDepth = kernel.shape.get(2)       // = inTensor.shape.get(2)

        // Iterate through the number of filters
        for (filter in 0 until kernel.shape.get(3)-1){
            // Iterate through all possible positions of the filter
            for (input_row in 0 until inputHeight-filterHeight+1){
                for(input_col in 0 until inputWidth-filterWidth+1){
                    //Apply the filter for each element in x, y and z direction
                    var y_i = 0f
                    for(filter_row in 0 until filterHeight){
                        for (filter_col in 0 until filter_row){
                            for (channel in 0 until imageDepth){
                                y_i += inTensor.get(input_row, input_col, channel) *
                                        kernel.get(filter_row, filter_col, channel, filter)
                            }
                        }
                    }
                    // Write result of filter application to the outTensor
                    outTensor.set(y_i, input_row, input_col, filter)
                }
            }
        }

    }

    /**
     * Adds the bias to each filter output
     */
    private fun addBiasPerFilter(inAndOutTensor: Tensor, bias: Tensor){
        for (filter in 0 until kernel.shape.get(3)){
            for(row in 0 until inAndOutTensor.shape.get(0)){
                for (col in 0 until inAndOutTensor.shape.get(1)){
                    val y_i_new = inAndOutTensor.get(row, col, filter) + bias.get(filter)
                    inAndOutTensor.set(y_i_new, row, col, filter)
                }
            }
        }
    }
}