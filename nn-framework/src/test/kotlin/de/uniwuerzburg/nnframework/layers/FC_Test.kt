package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import org.testng.Assert
import org.testng.annotations.Test

/**
 * @author vb
 */

class FC_Test {

    private val EPSILON = 0.00001f

    private val fc_layer = FullyConnectedLayer(Shape(intArrayOf(1,2)), Shape(intArrayOf(1,3)))
    private val out_tensors = listOf<Tensor>(Tensor(Shape(intArrayOf(1,3)), FloatArray(3)),
                                             Tensor(Shape(intArrayOf(1,3)), FloatArray(3)))
    private val in_tensors = listOf<Tensor>(
            Tensor(Shape(intArrayOf(1,2)), IntRange(0,1).toList().map { i: Int -> i.toFloat() }.toFloatArray()),
            Tensor(Shape(intArrayOf(1,2)), IntRange(6,7).toList().map { i: Int -> i.toFloat() }.toFloatArray()))

    @Test
    fun testForward() {
        fc_layer.setWeightsForTesting(  Tensor(Shape(intArrayOf(1,3)), floatArrayOf(0.55f, 0.96f, 0.93f)),
                Tensor(Shape(intArrayOf(2,3)), floatArrayOf(-0.71f, -0.84f,
                        0.62f, -0.54f,
                        -0.92f, -0.23f)))
        fc_layer.forward(in_tensors, out_tensors)
        val out_tensor1 = out_tensors.get(0)
        val out_tensor2 = out_tensors.get(1)

        Assert.assertEquals(out_tensor1.get(0), -0.29f, EPSILON)
        Assert.assertEquals(out_tensor1.get(1), 0.42f, EPSILON)
        Assert.assertEquals(out_tensor1.get(2), 0.7f, EPSILON)

        Assert.assertEquals(out_tensor2.get(0), -9.59f, EPSILON)
        Assert.assertEquals(out_tensor2.get(1), 0.9f, EPSILON)
        Assert.assertEquals(out_tensor2.get(2), -6.2f, EPSILON)

    }

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
}