package de.uniwuerzburg.nnframework.data

import org.testng.Assert
import org.testng.annotations.Test

class TensorUtilsTest {
    private val vector = Tensor(Shape(intArrayOf(3)), createFloatArray(0..2))
    private val matrix = Tensor(Shape(intArrayOf(4, 3)), createFloatArray(0..11))

    @Test
    fun testMult() {
        val result = mult(matrix, vector)

        Assert.assertEquals(result.shape.dimensions, 1)
        Assert.assertEquals(result.shape.get(0), 4)

        Assert.assertEquals(result.get(0), 20f)
        Assert.assertEquals(result.get(1), 23f)
        Assert.assertEquals(result.get(2), 26f)
        Assert.assertEquals(result.get(3), 29f)
    }

    @Test(expectedExceptions = [IllegalArgumentException::class])
    fun testMultIllegalShapes() {
        mult(vector, matrix)
    }

    @Test
    fun testAdd() {
        val result = add(vector, vector)

        Assert.assertEquals(result.shape.dimensions, 1)
        Assert.assertEquals(result.shape.get(0), 3)

        Assert.assertEquals(result.get(0), 0f)
        Assert.assertEquals(result.get(1), 2f)
        Assert.assertEquals(result.get(2), 4f)
    }

    @Test(expectedExceptions = [IllegalArgumentException::class])
    fun testAddIllegalShapes() {
        add(vector, matrix)
    }

    private fun createFloatArray(range: IntRange): FloatArray = range.toList().map { i: Int -> i.toFloat() }.toFloatArray()
}