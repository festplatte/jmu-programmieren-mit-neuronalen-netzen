package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.data.initializeWeights
import de.uniwuerzburg.nnframework.data.printTensor
import org.testng.Assert
import org.testng.annotations.Test

/**
 * @author vb
 */

class Conv2D_Test {

    private val EPSILON = 0.00001f

    private val conv2D_layer = Conv2DLayer( inputShape = Shape(intArrayOf(3,3,2)),
                                            outShape = Shape(intArrayOf(2,2,2)),
                                            filterShape = Shape(intArrayOf(2,2,2)),
                                            numOfFilters = 2)
    private val out_tensors = listOf<Tensor>(Tensor(Shape(intArrayOf(2,2,2))))
    private val in_tensors = listOf<Tensor>(
            Tensor(Shape(intArrayOf(3,3,2)), IntRange(1,18).toList().map { i: Int -> i.toFloat() }.toFloatArray()))

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

    /*

    @Test
    fun testBackward() {
        fc_layer.setWeightsForTesting(  Tensor(Shape(intArrayOf(1,3)), floatArrayOf(0.55f, 0.96f, 0.93f)),
                Tensor(Shape(intArrayOf(2,3)), floatArrayOf(-0.71f, -0.84f,
                        0.62f, -0.54f,
                        -0.92f, -0.23f)))
        // Set artificial delta values for the deltas of the out tensors
        out_tensors.get(0).setDeltas(floatArrayOf(-0.5f, 0.33f, 1.7f))
        out_tensors.get(1).setDeltas(floatArrayOf(-1f, 2.66f, -2.1f))


        fc_layer.backward(out_tensors, in_tensors)
        val in_tensor1 = in_tensors.get(0)
        val in_tensor2 = in_tensors.get(1)

        Assert.assertEquals(in_tensor1.deltas[0], -1.0044f, EPSILON)
        Assert.assertEquals(in_tensor1.deltas[1], -0.1492f, EPSILON)

        Assert.assertEquals(in_tensor2.deltas[0], 4.2912f, EPSILON)
        Assert.assertEquals(in_tensor2.deltas[1], -0.1134f, EPSILON)

    }

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