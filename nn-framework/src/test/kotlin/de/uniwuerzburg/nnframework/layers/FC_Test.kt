package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import org.testng.Assert
import org.testng.annotations.Test
import de.uniwuerzburg.nnframework.data.printTensor

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

    private fun createFloatArray(range: IntRange): FloatArray = range.toList().map { i: Int -> i.toFloat() }.toFloatArray()
}