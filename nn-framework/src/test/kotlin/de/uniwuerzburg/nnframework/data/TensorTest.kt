package de.uniwuerzburg.nnframework.data

import org.testng.Assert
import org.testng.annotations.Test

class TensorTest {
    private val vector = Tensor(Shape(intArrayOf(3)), createFloatArray(0..2))
    private val matrix = Tensor(Shape(intArrayOf(4,3)), createFloatArray(0..11))
    private val matrix3d = Tensor(Shape(intArrayOf(4,3,2)), createFloatArray(0..23))

    @Test
    fun testGet() {
        Assert.assertEquals(vector.get(1), 1f)
        Assert.assertEquals(matrix.get(1,2), 9f)
        Assert.assertEquals(matrix3d.get(2,1,1), 18f)
    }

    @Test
    fun testMult() {
        val result = matrix.mult(vector)

        Assert.assertEquals(result.shape.dimensions, 1)
        Assert.assertEquals(result.shape.get(0), 4)

        Assert.assertEquals(result.get(0), 20f)
        Assert.assertEquals(result.get(1), 23f)
        Assert.assertEquals(result.get(2), 26f)
        Assert.assertEquals(result.get(3), 29f)
    }

    @Test
    fun testAdd() {
        val result = vector.add(vector)

        Assert.assertEquals(result.shape.dimensions, 1)
        Assert.assertEquals(result.shape.get(0), 3)

        Assert.assertEquals(result.get(0), 0f)
        Assert.assertEquals(result.get(1), 2f)
        Assert.assertEquals(result.get(2), 4f)
    }

    fun createFloatArray(range: IntRange): FloatArray = range.toList().map { i: Int -> i.toFloat() }.toFloatArray()
}