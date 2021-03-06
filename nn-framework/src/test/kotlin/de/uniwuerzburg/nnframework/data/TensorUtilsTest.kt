package de.uniwuerzburg.nnframework.data

import org.testng.Assert
import org.testng.annotations.Test

class TensorUtilsTest {
    private val EPSILON = 0.00001f

    private val vector = Tensor(Shape(intArrayOf(3)), createFloatArray(0..2))
    private val matrix = Tensor(Shape(intArrayOf(4, 3)), createFloatArray(0..11))

    private val row_vector_tranposition = Tensor(Shape(intArrayOf(1,3)), floatArrayOf(-0.29f, 0.42f, 0.7f))
    private val col_vector_transposition = Tensor(Shape(intArrayOf(2)), floatArrayOf(-0.13f, 0.55f))
    private val matrix__tranposition = Tensor(Shape(intArrayOf(2,3)), floatArrayOf(  -0.71f, -0.84f,
                                                                                    0.62f, -0.54f,
                                                                                    -0.92f, -0.23f))



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

    @Test
    fun testMultWithTransposeFirst() {
        // (2x1)^T*(2x3)
        val result1 = multAndTransposeFirst(col_vector_transposition, matrix__tranposition)

        Assert.assertEquals(result1.shape.dimensions, 2)
        Assert.assertEquals(result1.shape.get(0), 1)
        Assert.assertEquals(result1.shape.get(1), 3)

        Assert.assertEquals(result1.get(0,0), -0.3697f, EPSILON)
        Assert.assertEquals(result1.get(0,1), -0.3776f, EPSILON)
        Assert.assertEquals(result1.get(0,2), -0.0069f, EPSILON)

        // (2x3)^T*(2x1)
        val result2 = multAndTransposeFirst(matrix__tranposition,col_vector_transposition)
        //Assert.assertEquals(result2.shape.dimensions, 1)
        Assert.assertEquals(result2.shape.get(0), 3)

        Assert.assertEquals(result2.get(0), -0.3697f, EPSILON)
        Assert.assertEquals(result2.get(1), -0.3776f, EPSILON)
        Assert.assertEquals(result2.get(2), -0.0069f, EPSILON)

    }

    @Test
    fun testMultWithTransposeFirstAndSumUp() {
        // (2x1)^T*(2x3)
        val result = Tensor(Shape(intArrayOf(1,3)), FloatArray(3))
        result.deltas = createFloatArray(0..2)
        multAndTransposeFirst(col_vector_transposition, matrix__tranposition, result,
                tensorA_useDeltas = false, tensorB_useDeltas = false, outTensor_useDeltas = true, outTensor_sumUp = true)

        Assert.assertEquals(result.shape.dimensions, 2)
        Assert.assertEquals(result.shape.get(0), 1)
        Assert.assertEquals(result.shape.get(1), 3)

        Assert.assertEquals(result.deltas[0], -0.3697f + 0f, EPSILON)
        Assert.assertEquals(result.deltas[1], -0.3776f + 1f, EPSILON)
        Assert.assertEquals(result.deltas[2], -0.0069f + 2f, EPSILON)


    }

    @Test
    fun testMultWithTransposeSecond() {
        // (1x3)*(2x3)^T
        val result1 = multAndTransposeSecond(row_vector_tranposition, matrix__tranposition)

        Assert.assertEquals(result1.shape.dimensions, 2)
        Assert.assertEquals(result1.shape.get(0), 1)
        Assert.assertEquals(result1.shape.get(1), 2)

        Assert.assertEquals(result1.get(0,0), -0.1777f, EPSILON)
        Assert.assertEquals(result1.get(0,1), -0.1442f, EPSILON)

        // (2x3)*(1x3)^T
        val result2 = multAndTransposeSecond(matrix__tranposition,row_vector_tranposition)
        //Assert.assertEquals(result2.shape.dimensions, 1)
        Assert.assertEquals(result2.shape.get(0), 2)

        Assert.assertEquals(result2.get(0), -0.1777f, EPSILON)
        Assert.assertEquals(result2.get(1), -0.1442f, EPSILON)

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