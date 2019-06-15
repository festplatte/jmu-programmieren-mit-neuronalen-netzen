package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import org.testng.Assert
import org.testng.annotations.Test
import de.uniwuerzburg.nnframework.data.printTensor

class FC_Test {
    private val fc_layer = FullyConnectedLayer(Shape(intArrayOf(2)), Shape(intArrayOf(3)))
    private val out_tensors = listOf<Tensor>(Tensor(Shape(intArrayOf(3)), FloatArray(3)))
    private val in_tensors = listOf<Tensor>(Tensor(Shape(intArrayOf(2)), IntRange(0,1).toList().map { i: Int -> i.toFloat() }.toFloatArray()))

    @Test
    fun testForward() {
        fc_layer.setWeightsForTesting(  Tensor(Shape(intArrayOf(3)), floatArrayOf(0.55f, 0.96f, 0.93f)),
                Tensor(Shape(intArrayOf(2,3)), floatArrayOf(-0.71f, -0.84f,
                        0.62f, -0.54f,
                        -0.92f, 0.23f)))
        fc_layer.forward(in_tensors, out_tensors)
        val out_tensor = out_tensors.get(0)
        println("Result:")
        printTensor(out_tensor)

        Assert.assertEquals(out_tensor.get(0), -0.29f)
        Assert.assertEquals(out_tensor.get(1), 0.42f)
        Assert.assertEquals(out_tensor.get(2), 0.7f)
    }

    private fun createFloatArray(range: IntRange): FloatArray = range.toList().map { i: Int -> i.toFloat() }.toFloatArray()
}