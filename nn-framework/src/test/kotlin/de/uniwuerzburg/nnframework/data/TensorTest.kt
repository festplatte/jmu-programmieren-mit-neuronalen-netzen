package de.uniwuerzburg.nnframework.data

import org.testng.Assert
import org.testng.annotations.Test

class TensorTest {
    private val vector = Tensor(Shape(intArrayOf(3)), createFloatArray(0..2))
    private val matrix = Tensor(Shape(intArrayOf(4, 3)), createFloatArray(0..11))
    private val matrix3d = Tensor(Shape(intArrayOf(4, 3, 2)), createFloatArray(0..23))

    @Test
    fun testGet() {
        Assert.assertEquals(vector.get(1), 1f)
        Assert.assertEquals(matrix.get(1, 2), 9f)
        Assert.assertEquals(matrix3d.get(2, 1, 1), 18f)
    }

    @Test(expectedExceptions = [IllegalArgumentException::class])
    fun testTensorWithInvalidShape() {
        Tensor(Shape(intArrayOf(4, 3)), createFloatArray(0..12))
    }

    private fun createFloatArray(range: IntRange): FloatArray = range.toList().map { i: Int -> i.toFloat() }.toFloatArray()
}