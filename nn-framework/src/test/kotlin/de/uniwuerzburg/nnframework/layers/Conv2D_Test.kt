package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.data.printTensor
import org.testng.Assert
import org.testng.annotations.Test

/**
 * @author vb
 */

class Conv2D_Test {

    private val EPSILON = 0.00001f

    // For [2,2,2,2]-shaped kernel
    private val conv2D_layer = Conv2DLayer( inputShape = Shape(intArrayOf(3,3,2)),
                                            outShape = Shape(intArrayOf(2,2,2)),
                                            filterShape = Shape(intArrayOf(2,2,2)),
                                            numOfFilters = 2)
    private val out_tensors = listOf<Tensor>(Tensor(Shape(intArrayOf(2,2,2))))
    private val in_tensors = listOf<Tensor>(
            Tensor(Shape(intArrayOf(3,3,2)), IntRange(1,18).toList().map { i: Int -> i.toFloat() }.toFloatArray()))

    // For [2,3,2,2]-shaped kernel
    private val conv2D_layer2 = Conv2DLayer( inputShape = Shape(intArrayOf(5,5,2)),
            outShape = Shape(intArrayOf(4,3,2)),
            filterShape = Shape(intArrayOf(2,3,2)),
            numOfFilters = 2)

    @Test
    fun testForward() {
        //Convolution without bias
        conv2D_layer.setWeightsForTesting(  bias = Tensor(Shape(intArrayOf(2)), floatArrayOf(0f,0f)),
                kernel = Tensor(Shape(intArrayOf(2,2,2,2)),
                        floatArrayOf(0.74f, -0.15f, -0.55f, -0.69f,
                                0.04f, 0.87f, 0.87f, -0.48f,
                                -0.71f, 0.69f, -0.64f, 0.76f,
                                0.98f, -0.97f, 0.7f, 0.26f)))

        conv2D_layer.forward(in_tensors,out_tensors)
        val out_tensor = out_tensors.get(0)

        Assert.assertEquals(out_tensor.get(0,0,0), 9.35f, EPSILON)
        Assert.assertEquals(out_tensor.get(1,0,0), 10f, EPSILON)
        Assert.assertEquals(out_tensor.get(0,1,0), 11.3f, EPSILON)
        Assert.assertEquals(out_tensor.get(1,1,0), 11.95f, EPSILON)

        Assert.assertEquals(out_tensor.get(0,0,1), 13.78f, EPSILON)
        Assert.assertEquals(out_tensor.get(1,0,1), 14.85f, EPSILON)
        Assert.assertEquals(out_tensor.get(0,1,1), 16.99f, EPSILON)
        Assert.assertEquals(out_tensor.get(1,1,1), 18.06f, EPSILON)


        //Convolution with bias
        conv2D_layer.setWeightsForTesting(  bias = Tensor(Shape(intArrayOf(2)), floatArrayOf(-0.12f,0.55f)),
                kernel = Tensor(Shape(intArrayOf(2,2,2,2)),
                        floatArrayOf(0.74f, -0.15f, -0.55f, -0.69f,
                                0.04f, 0.87f, 0.87f, -0.48f,
                                -0.71f, 0.69f, -0.64f, 0.76f,
                                0.98f, -0.97f, 0.7f, 0.26f)))

        conv2D_layer.forward(in_tensors,out_tensors)
        val out_tensor_with_bias = out_tensors.get(0)

        Assert.assertEquals(out_tensor_with_bias.get(0,0,0), 9.35f - 0.12f, EPSILON)
        Assert.assertEquals(out_tensor_with_bias.get(1,0,0), 10f - 0.12f, EPSILON)
        Assert.assertEquals(out_tensor_with_bias.get(0,1,0), 11.3f - 0.12f, EPSILON)
        Assert.assertEquals(out_tensor_with_bias.get(1,1,0), 11.95f - 0.12f, EPSILON)

        Assert.assertEquals(out_tensor_with_bias.get(0,0,1), 13.78f + 0.55f, EPSILON)
        Assert.assertEquals(out_tensor_with_bias.get(1,0,1), 14.85f + 0.55f, EPSILON)
        Assert.assertEquals(out_tensor_with_bias.get(0,1,1), 16.99f + 0.55f, EPSILON)
        Assert.assertEquals(out_tensor_with_bias.get(1,1,1), 18.06f + 0.55f, EPSILON)
    }

    @Test
    fun testTransposeDepthAndAmountOfFilters(){
        conv2D_layer.setWeightsForTesting(  bias = Tensor(Shape(intArrayOf(2)), floatArrayOf(0f,0f)),
                kernel = Tensor(Shape(intArrayOf(2,2,2,2)),
                        floatArrayOf(0.74f, -0.15f, -0.55f, -0.69f,
                                0.04f, 0.87f, 0.87f, -0.48f,
                                -0.71f, 0.69f, -0.64f, 0.76f,
                                0.98f, -0.97f, 0.7f, 0.26f)))

        Assert.assertEquals(conv2D_layer.transposedKernel.get(0,0,0,0), 0.74f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(1,0,0,0), -0.15f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(0,1,0,0), -0.55f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(1,1,0,0), -0.69f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(0,0,1,0), -0.71f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(1,0,1,0), 0.69f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(0,1,1,0), -0.64f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(1,1,1,0), 0.76f, EPSILON)

        Assert.assertEquals(conv2D_layer.transposedKernel.get(0,0,0,1), 0.04f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(1,0,0,1), 0.87f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(0,1,0,1), 0.87f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(1,1,0,1), -0.48f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(0,0,1,1), 0.98f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(1,0,1,1), -0.97f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(0,1,1,1), 0.7f, EPSILON)
        Assert.assertEquals(conv2D_layer.transposedKernel.get(1,1,1,1), 0.26f, EPSILON)
    }

    @Test
    fun testTransposeDepthAndAmountOfFiltersAndRotateBy180Degrees(){
        // Test vor [2,2,2,2]-shaped kernel
        conv2D_layer.setWeightsForTesting(  bias = Tensor(Shape(intArrayOf(2)), floatArrayOf(0f,0f)),
                kernel = Tensor(Shape(intArrayOf(2,2,2,2)),
                        floatArrayOf(0.74f, -0.15f, -0.55f, -0.69f,
                                0.04f, 0.87f, 0.87f, -0.48f,
                                -0.71f, 0.69f, -0.64f, 0.76f,
                                0.98f, -0.97f, 0.7f, 0.26f)))

        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(0,0,0,0), -0.69f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(1,0,0,0), -0.55f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(0,1,0,0), -0.15f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(1,1,0,0), 0.74f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(0,0,1,0), 0.76f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(1,0,1,0), -0.64f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(0,1,1,0), 0.69f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(1,1,1,0), -0.71f, EPSILON)

        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(0,0,0,1), -0.48f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(1,0,0,1), 0.87f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(0,1,0,1), 0.87f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(1,1,0,1), 0.04f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(0,0,1,1), 0.26f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(1,0,1,1), 0.7f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(0,1,1,1), -0.97f, EPSILON)
        Assert.assertEquals(conv2D_layer.rotatedTransposedKernel.get(1,1,1,1), 0.98f, EPSILON)

        // Test vor [2,3,2,2]-shaped kernel
        conv2D_layer2.setWeightsForTesting(  bias = Tensor(Shape(intArrayOf(2)), floatArrayOf(0f,0f)),
                kernel = Tensor(Shape(intArrayOf(2,3,2,2)),
                        floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f,
                                     7f, 8f, 9f, 10f, 11f, 12f,
                                     13f, 24f, 15f, 16f, 17f, 18f,
                                     19f, 20f, 21f, 22f, 23f, 24f)))

        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,0,0,0), 6f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,0,0,0), 5f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,1,0,0), 4f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,1,0,0), 3f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,2,0,0), 2f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,2,0,0), 1f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,0,1,0), 18f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,0,1,0), 17f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,1,1,0), 16f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,1,1,0), 15f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,2,1,0), 24f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,2,1,0), 13f)

        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,0,0,1), 12f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,0,0,1), 11f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,1,0,1), 10f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,1,0,1), 9f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,2,0,1), 8f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,2,0,1), 7f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,0,1,1), 24f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,0,1,1), 23f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,1,1,1), 22f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,1,1,1), 21f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(0,2,1,1), 20f)
        Assert.assertEquals(conv2D_layer2.rotatedTransposedKernel.get(1,2,1,1), 19f)

    }

    @Test
    fun testFullConvolve(){
        val inTensor: Tensor = in_tensors.get(0)
        val kernel = Tensor(Shape(intArrayOf(2,2,2,2)),
                                    IntRange(1,16).toList().map { i: Int -> i.toFloat() }.toFloatArray())
        val outTensor = Tensor(Shape(intArrayOf(4,4,2)))

        conv2D_layer.executeFullConvolve(inTensor, kernel, outTensor)

        // Filter 1
        Assert.assertEquals(outTensor.get(0,0,0), 84f)
        Assert.assertEquals(outTensor.get(1,0,0), 169f)
        Assert.assertEquals(outTensor.get(2,0,0), 191f)
        Assert.assertEquals(outTensor.get(3,0,0), 93f)

        Assert.assertEquals(outTensor.get(0,1,0), 182f)
        Assert.assertEquals(outTensor.get(1,1,0), 356f)
        Assert.assertEquals(outTensor.get(2,1,0), 392f)
        Assert.assertEquals(outTensor.get(3,1,0), 186f)

        Assert.assertEquals(outTensor.get(0,2,0), 242f)
        Assert.assertEquals(outTensor.get(1,2,0), 464f)
        Assert.assertEquals(outTensor.get(2,2,0), 500f)
        Assert.assertEquals(outTensor.get(3,2,0), 234f)

        Assert.assertEquals(outTensor.get(0,3,0), 110f)
        Assert.assertEquals(outTensor.get(1,3,0), 205f)
        Assert.assertEquals(outTensor.get(2,3,0), 219f)
        Assert.assertEquals(outTensor.get(3,3,0), 99f)

        // Filter 2
        Assert.assertEquals(outTensor.get(0,0,1), 172f)
        Assert.assertEquals(outTensor.get(1,0,1), 361f)
        Assert.assertEquals(outTensor.get(2,0,1), 415f)
        Assert.assertEquals(outTensor.get(3,0,1), 213f)

        Assert.assertEquals(outTensor.get(0,1,1), 406f)
        Assert.assertEquals(outTensor.get(1,1,1), 836f)
        Assert.assertEquals(outTensor.get(2,1,1), 936f)
        Assert.assertEquals(outTensor.get(3,1,1), 474f)

        Assert.assertEquals(outTensor.get(0,2,1), 562f)
        Assert.assertEquals(outTensor.get(1,2,1), 1136f)
        Assert.assertEquals(outTensor.get(2,2,1), 1236f)
        Assert.assertEquals(outTensor.get(3,2,1), 618f)

        Assert.assertEquals(outTensor.get(0,3,1), 294f)
        Assert.assertEquals(outTensor.get(1,3,1), 589f)
        Assert.assertEquals(outTensor.get(2,3,1), 635f)
        Assert.assertEquals(outTensor.get(3,3,1), 315f)
    }



    @Test
    fun testBackward() {
        conv2D_layer.setWeightsForTesting(  bias = Tensor(Shape(intArrayOf(2)), floatArrayOf(0f,0f)),
                kernel = Tensor(Shape(intArrayOf(2,2,2,2)),
                        floatArrayOf(0.74f, -0.15f, -0.55f, -0.69f,
                                0.04f, 0.87f, 0.87f, -0.48f,
                                -0.71f, 0.69f, -0.64f, 0.76f,
                                0.98f, -0.97f, 0.7f, 0.26f)))

        // Set artificial delta values for the deltas of the out tensors
        out_tensors.get(0).setDeltas(floatArrayOf(0.79f, -0.39f, 0.88f, 0.46f, 0.75f, -0.99f, -0.28f, 0.16f))

        conv2D_layer.backward(out_tensors, in_tensors)
        val in_tensor = in_tensors.get(0)

        /**
         * Expected results:
         *
         * Channel 1:
         * [[ 0.0521 -0.0645 -0.3048]
         * [ 0.8133  0.7746 -1.1754]
         * [-0.6246 -0.4419 -0.1958]]
         *
         * Channel 2:
         * [[ 0.7666  0.9731  0.5696]
         * [-1.026   -0.0041  0.017 ]
         * [ 0.621    0.1748 -0.1792]]
         */

        //Channel 1
        Assert.assertEquals(in_tensor.deltas[0], 0.0521f, EPSILON)       //Delta x_0
        Assert.assertEquals(in_tensor.deltas[1], 0.8133f, EPSILON)
        Assert.assertEquals(in_tensor.deltas[2], -0.6246f, EPSILON)

        Assert.assertEquals(in_tensor.deltas[3], -0.0645f, EPSILON)       //Delta x_3
        Assert.assertEquals(in_tensor.deltas[4], 0.7746f, EPSILON)
        Assert.assertEquals(in_tensor.deltas[5], -0.4419f, EPSILON)

        Assert.assertEquals(in_tensor.deltas[6], -0.3048f, EPSILON)       //Delta x_6
        Assert.assertEquals(in_tensor.deltas[7], -1.1754f, EPSILON)
        Assert.assertEquals(in_tensor.deltas[8], -0.1958f, EPSILON)

        //Channel 2
        Assert.assertEquals(in_tensor.deltas[9], 0.7666f, EPSILON)       //Delta x_9
        Assert.assertEquals(in_tensor.deltas[10], -1.026f, EPSILON)
        Assert.assertEquals(in_tensor.deltas[11], 0.621f, EPSILON)

        Assert.assertEquals(in_tensor.deltas[12], 0.9731f, EPSILON)      //Delta x_12
        Assert.assertEquals(in_tensor.deltas[13], -0.0041f, EPSILON)
        Assert.assertEquals(in_tensor.deltas[14], 0.1748f, EPSILON)

        Assert.assertEquals(in_tensor.deltas[15], 0.5696f, EPSILON)      //Delta x_15
        Assert.assertEquals(in_tensor.deltas[16], 0.017f, EPSILON)
        Assert.assertEquals(in_tensor.deltas[17], -0.1792f, EPSILON)


    }

    /*
    @Test
    fun testCalculateDeltaWeights(){
        fc_layer.setWeightsForTesting(  Tensor(Shape(intArrayOf(1,3)), floatArrayOf(0.55f, 0.96f, 0.93f)),
                Tensor(Shape(intArrayOf(2,3)), floatArrayOf(-0.71f, -0.84f,
                        0.62f, -0.54f,
                        -0.92f, -0.23f)))
        // Set artificial delta values for the deltas of the out tensors
        out_tensors.get(0).setDeltas(floatArrayOf(-0.5f, 0.33f, 1.7f))
        out_tensors.get(1).setDeltas(floatArrayOf(-1f, 2.66f, -2.1f))

        fc_layer.calculateDeltaWeights(out_tensors, in_tensors)
        val bias = fc_layer.getBias
        val weights = fc_layer.getWeights

        Assert.assertEquals(bias.getDelta(0), -1.5f, EPSILON)
        Assert.assertEquals(bias.getDelta(1), 2.99f, EPSILON)
        Assert.assertEquals(bias.getDelta(2), -0.4f, EPSILON)

        Assert.assertEquals(weights.getDelta(0, 0), -6f, EPSILON)
        Assert.assertEquals(weights.getDelta(0, 1), 15.96f, EPSILON)
        Assert.assertEquals(weights.getDelta(0, 2), -12.6f, EPSILON)
        Assert.assertEquals(weights.getDelta(1, 0), -7.5f, EPSILON)
        Assert.assertEquals(weights.getDelta(1, 1), 18.95f, EPSILON)
        Assert.assertEquals(weights.getDelta(1, 2), -13f, EPSILON)
    }
    */

}