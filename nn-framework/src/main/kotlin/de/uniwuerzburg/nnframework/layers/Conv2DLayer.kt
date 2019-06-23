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

    // The rotated, transposed kernel is needed for the backward() method
    // Actually, one Tensor that is sequentially transformed would be sufficient,
    // but it is more clearly that always both operations are needed when splitting it into two variables
    var transposedKernel: Tensor = Tensor(Shape(intArrayOf(0)))  //Dummy value, will get overwritten during init()
    var rotatedTransposedKernel: Tensor = Tensor(Shape(intArrayOf(0)))  //Dummy value, will get overwritten during init()

    init {
        // Check validty of parameters
        if(inputShape.dimensions >3){
            throw IllegalArgumentException("The convoltion layer can not deal with input dimensions greater than 3!")
        }

        if (inputShape.get(2)!= filterShape.get(2)) {
            throw IllegalArgumentException("The filter depth must match the input depth!")
        }

        val calculatedOutShape = Shape(intArrayOf(inputShape.get(0) - filterShape.get(0) + 1 ,
                                                  inputShape.get(1) - filterShape.get(1) + 1 ,
                                                  numOfFilters))

        if(!outShape.equals(calculatedOutShape)){
            throw IllegalArgumentException("The output shape must match the input shape and the filter shape!")
        }

        // The kernel tensor has one dimension more that the filter Shape, the additional dimension corresponds to the
        // number of filters dimensions.add(tensorB.shape.get(1)
        val dimensions = filterShape.axis.clone().toMutableList()
        dimensions.add(numOfFilters)
        val kernel_tensor_shape = Shape(dimensions.toIntArray())
        kernel = Tensor(kernel_tensor_shape)

        transposedKernel = Tensor(Shape(intArrayOf( kernel.shape.get(0), kernel.shape.get(1),
                                                    kernel.shape.get(3), kernel.shape.get(2))))
        rotatedTransposedKernel = Tensor(Shape(transposedKernel.shape.axis.clone()))


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
    * deltaX = deltaY * rot_180 ( trans_0,1,3,2 (F)) with
    *       deltaX: delta of the input tensor
    *       F: kernel tensor
    *       deltaY: delta of the output tensor
    *       *: full convolution operator for images with a depth
    */
    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        //Prepare the filter for the backward pass
        transposeDepthAndAmountOfFilters(kernel, transposedKernel)
        rotateBy180Degrees(transposedKernel, rotatedTransposedKernel)

        for (i in inTensors.indices){
            val inTensor = inTensors.get(i)
            val outTensor = outTensors.get(i)
            //use deltas of the outTensor and write the result to the deltas of the inTensor
            fullConvolve(inTensor = outTensor, kernel = rotatedTransposedKernel, outTensor = inTensor,
                         inTensor_useDeltas = true, outTensor_useDeltas = true)
        }
    }

    /*
    * The calculateDeltaWeights function calculates the delta weights by
    * using the elements of the inTensors and the deltas of the outTensors
    * DeltaBias = Sum of all elements of DeltaY of the according channel
    * DeltaFilter = X *ch DeltaY        (*ch: channel-wise convolution)
    *
    * If the list contains more than one element, the update values are summed up
    */
    override fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>) {

        val biasDeltas = FloatArray(bias.shape.get(0))

        for (i in inTensors.indices){
            val inTensor = inTensors.get(i)
            val outTensor = outTensors.get(i)

            // Calculate the delta weights for the bias and sum them up
            for (bias_i in 0 until bias.shape.get(0)){
                for(row in 0 until outTensor.shape.get(0)){
                    for (col in 0 until outTensor.shape.get(1)){
                        biasDeltas[bias_i] += outTensor.getDelta(row, col, bias_i)
                    }
                }
            }

            //Calculate the delta weights for the filters and sum them up
            channelwiseConvolve(inTensor = inTensor, convolvTensor = outTensor, outTensor = kernel,
                                convolvTensor_useDeltas = true, outTensor_useDeltas = true,
                                outTensor_sumUp = true)
        }

        // The bias deltas still net to be written
        bias.deltas = biasDeltas
    }

    override fun updateWeights(updater: (value: Float, delta: Float) -> Float) {
        // TODO("not implemented") To change body of created functions use File | Settings | File Templates.
    }

    /**
     * This function writes the result of applying the convolution operator without padding for images with depth
     * to the elements of the outTensor
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
        for (filter in 0 until kernel.shape.get(3)){
            // Iterate through all possible positions of the filter
            for (input_row in 0 until inputHeight-filterHeight+1){
                for(input_col in 0 until inputWidth-filterWidth+1){
                    //Apply the filter for each element in x, y and z direction
                    var y_i = 0f
                    for(filter_row in 0 until filterHeight){
                        for (filter_col in 0 until filterWidth){
                            for (channel in 0 until imageDepth){
                                y_i += inTensor.get(input_row + filter_row, input_col+filter_col, channel) *
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

    /**
     * This function rotates the filters in x-y-axis by 180 degrees and returns a new, rotated kernel
     * to the outRotatedKernel parameter
     */
    private fun rotateBy180Degrees(kernel: Tensor, outRotatedKernel: Tensor){
        for (filter in 0 until kernel.shape.get(3)){
            for (channel in 0 until kernel.shape.get(2)){
                for(kernel_row in 0 until kernel.shape.get(0)){
                    for(kernel_col in 0 until kernel.shape.get(1)){
                        val current_val = kernel.get(kernel_row, kernel_col, channel, filter)
                        val new_row = kernel.shape.get(0) - kernel_row - 1
                        val new_col = kernel.shape.get(1) - kernel_col - 1
                        outRotatedKernel.set(current_val, new_row, new_col, channel, filter)
                    }
                }

            }
        }
    }

    /**
     * This functions transposes the depth of a filter with the amount of filters and writes the new, transposed kernel
     * to the outTransposedKernel parameter
     * [x,y,c,f] = x,y,f,c
     */
    private fun transposeDepthAndAmountOfFilters(kernel: Tensor, outTransposedKernel: Tensor){

        for (filter in 0 until kernel.shape.get(3)){
            for (channel in 0 until kernel.shape.get(2)){
                for(kernel_row in 0 until kernel.shape.get(0)){
                    for(kernel_col in 0 until kernel.shape.get(1)){
                        val value = kernel.get(kernel_row, kernel_col, channel, filter)
                        outTransposedKernel.set(value, kernel_row, kernel_col, filter, channel)
                    }
                }

            }
        }
    }

    /**
     * This function writes the result of applying the full convolution operator (full padding) for images with depth
     * to the elements or the deltas of the outTensor (depending on the outTensor_useDeltas parameter)
     * A 2-d convolution ‘convolves’ along two spatial dimensions.
     * It has a really small kernel, essentially a window of pixel values, that slides along those two dimensions.
     * The rgb channel isn't handled as small window of depth, but obtained from beginning to end (first channel to last)
     * That is, even a convolution with a small spatial window of 1x1, which takes a single pixel spatially in the
     * width/height dimensions, would still take all 3 RGB channels
     */
    private fun fullConvolve(inTensor: Tensor, kernel: Tensor, outTensor: Tensor,
                             inTensor_useDeltas: Boolean = false, outTensor_useDeltas: Boolean = false){
        val inputHeight = inTensor.shape.get(0)
        val inputWidth = inTensor.shape.get(1)
        val filterHeight = kernel.shape.get(0)
        val filterWidth = kernel.shape.get(1)

        val paddingHeight = filterHeight - 1
        val paddingWidth = filterWidth -1
        val paddedInputHeight = inputHeight + 2 * paddingHeight
        val paddedInputWidth = inputWidth + 2 * paddingWidth

        val imageDepth = kernel.shape.get(2)       // = inTensor.shape.get(2)

        // Iterate through the number of filters
        for (filter in 0 until kernel.shape.get(3)){
            // Iterate through all possible positions of the filter (full padding!)
            // Every possible partial or complete superimposition of the kernel on the input map is taken into account
            for (input_row in 0 until paddedInputHeight - filterHeight + 1){
                for(input_col in 0 until paddedInputWidth - filterWidth + 1) {
                    //Apply the filter for each element in x, y and z direction
                    var y_i = 0f
                    for (filter_row in 0 until filterHeight) {
                        for (filter_col in 0 until filterWidth) {
                            for (channel in 0 until imageDepth) {
                                // Ignore everything outside the input map (would be 0 anyway)
                                // -> ignore input samples which are out of bound
                                val curr_row_in_orig_input = input_row + filter_row - paddingHeight
                                val curr_col_in_orig_input = input_col + filter_col - paddingWidth
                                if (curr_row_in_orig_input in 0 until inputHeight &&
                                        curr_col_in_orig_input in 0 until inputWidth) {
                                    if (inTensor_useDeltas) {
                                        y_i += inTensor.getDelta(curr_row_in_orig_input, curr_col_in_orig_input, channel) *
                                                kernel.get(filter_row, filter_col, channel, filter)
                                    } else {
                                        y_i += inTensor.get(curr_row_in_orig_input, curr_col_in_orig_input, channel) *
                                                kernel.get(filter_row, filter_col, channel, filter)
                                    }
                                }
                            }
                        }
                        // Write result of filter application to the outTensor
                        if (outTensor_useDeltas) {
                            outTensor.setDelta(y_i, input_row, input_col, filter)
                        } else {
                            outTensor.set(y_i, input_row, input_col, filter)
                        }
                    }
                }
            }
        }
    }

    /**
     * This function writes the result of applying the channelwise convolution operator without padding
     * for images with depth to the elements or the deltas of the outTensor
     * (depending on the outTensor_useDeltas parameter)
     * outTensor = inTensor *ch convolvTensor
     */
    fun channelwiseConvolve(inTensor: Tensor, convolvTensor: Tensor, outTensor: Tensor,
                                    convolvTensor_useDeltas: Boolean = false, outTensor_useDeltas: Boolean = false,
                                    outTensor_sumUp: Boolean = false){

        // Iterate through each filter of the outTensor
        for (filter in 0 until outTensor.shape.get(3)){
            // Iterate through the numbers of channels of the filter
            for (channel in 0 until outTensor.shape.get(2)){
                /*
                Example:
                Filter 1, Channel 1 => x[Channel 1] * delta_y[Channel 1]
                Filter 1, Channel 2 => x[Channel 2] * delta_y[Channel 1]
                Filter 1, Channel 3 => x[Channel 3] * delta_y[Channel 1]

                In General:
                Filter a, Channel b => x[Channel b] * delta_y[channel a]
                *
                 */
                simpleConvolveWithFixedChannels(inTensor = inTensor, convolvTensor = convolvTensor, outTensor = outTensor,
                        inTensor_channel = channel, convolvTensor_channel = filter,
                        outTensor_filter = filter, outTensor_channel = channel,
                        convolvTensor_useDeltas = convolvTensor_useDeltas, outTensor_useDeltas = outTensor_useDeltas,
                        outTensor_sumUp = outTensor_sumUp)
            }
        }

    }

    /**
     * This function writes the result of applying the convolution operator without padding and only for the specified
     * channel of the image and the filter with depth to the elements or the deltas of the specified outTensor channel
     * (depending on the outTensor_useDeltas parameter)
     */
    fun simpleConvolveWithFixedChannels(inTensor: Tensor, convolvTensor: Tensor, outTensor: Tensor,
                                    inTensor_channel: Int, convolvTensor_channel: Int,
                                    outTensor_filter: Int, outTensor_channel: Int,
                                    convolvTensor_useDeltas: Boolean = false, outTensor_useDeltas: Boolean = false,
                                    outTensor_sumUp: Boolean = false){

        val inputHeight = inTensor.shape.get(0)
        val inputWidth = inTensor.shape.get(1)
        val convolvTensorHeight = convolvTensor.shape.get(0)
        val convolvTensorWidth = convolvTensor.shape.get(1)

        // Iterate through all possible positions of the filter
        for (input_row in 0 until inputHeight-convolvTensorHeight+1){
            for(input_col in 0 until inputWidth-convolvTensorWidth+1){
                //Apply the filter for each element in x and y (there is no depth anymore!)
                var y_i = 0f
                for(filter_row in 0 until convolvTensorHeight){
                    for (filter_col in 0 until convolvTensorWidth){
                        if(convolvTensor_useDeltas){
                            y_i += inTensor.get(input_row + filter_row, input_col+filter_col, inTensor_channel) *
                                    convolvTensor.getDelta(filter_row, filter_col, convolvTensor_channel)
                        }else{
                            y_i += inTensor.get(input_row + filter_row, input_col+filter_col, inTensor_channel) *
                                    convolvTensor.get(filter_row, filter_col, convolvTensor_channel)
                        }

                    }
                }
                // Write result to the outTensor
                if(outTensor_useDeltas){
                    if(!outTensor_sumUp){
                        outTensor.setDelta(y_i, input_row, input_col, outTensor_channel, outTensor_filter)
                    }else{
                        val oldVal = outTensor.getDelta(input_row, input_col, outTensor_channel, outTensor_filter)
                        outTensor.setDelta(oldVal + y_i, input_row, input_col, outTensor_channel, outTensor_filter)
                    }

                }else{
                    if(!outTensor_sumUp){
                        outTensor.set(y_i, input_row, input_col, outTensor_channel, outTensor_filter)
                    }else{
                        val oldVal = outTensor.get(input_row, input_col, outTensor_channel, outTensor_filter)
                        outTensor.set(oldVal + y_i, input_row, input_col, outTensor_channel, outTensor_filter)
                    }
                }
            }
        }

    }

    /**
     * This function can be used to explicitly set the weights during testing
     * It also re-calculates the transposed kernel as well as the rotated, transposed kernel based on the new kernel
     */
    fun setWeightsForTesting(bias:Tensor, kernel:Tensor){
        this.bias = bias
        this.kernel = kernel
        transposeDepthAndAmountOfFilters(this.kernel, this.transposedKernel)
        rotateBy180Degrees(this.transposedKernel, this.rotatedTransposedKernel)
    }

    /**
     * This function is a wrapper around the fullConvolve function and can be used to test it
     */
    fun executeFullConvolve(inTensor: Tensor, kernel: Tensor, outTensor: Tensor){
        fullConvolve(inTensor, kernel, outTensor)
    }
}