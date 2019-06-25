package de.uniwuerzburg.nnframework

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.layers.*
import de.uniwuerzburg.nnframework.loss.CrossEntropyLoss
import org.testng.Assert
import org.testng.annotations.Test

class NetworkTest {

    private val EPSILON = 0.0001f

    /*val network = Network(ImageStringInputLayer(), listOf(
            FlatternLayer(Shape(intArrayOf(1, 784))),
            FullyConnectedLayer(Shape(intArrayOf(1, 784)), Shape(intArrayOf(1, 512))),
            SigmoidActivation(Shape(intArrayOf(1, 512))),
            FullyConnectedLayer(Shape(intArrayOf(1, 512)), Shape(intArrayOf(1, 10))),
            SoftmaxLayer(Shape(intArrayOf(1, 10)))
    ))*/
    //val trainer = SGDTrainer(1, 0.001f, 10, CrossEntropyLoss(), true, SGDFlavor.STOCHASTIC_GRADIENT_DESCENT)
    val fc1_weights = Tensor(Shape(intArrayOf(3,3)), floatArrayOf(-0.5057f, 0.3356f, -0.3485f, 0.3987f, 0.1673f, -0.4597f, -0.8943f, 0.8321f, -0.1121f))
    val fc2_weights = Tensor(Shape(intArrayOf(3,2)), floatArrayOf(0.4047f, -0.8192f, 0.3662f, 0.9563f, -0.1274f, -0.7253f))
    val fc1_bias = Tensor(Shape(intArrayOf(1, 3)), floatArrayOf(0f, 0f, 0f))
    val fc2_bias = Tensor(Shape(intArrayOf(1, 2)), floatArrayOf(0f, 0f))
    val input = listOf(Tensor(Shape(intArrayOf(1,3)), floatArrayOf(0.4183f, 0.5209f, 0.0291f)))
    val fc1_output = listOf(Tensor(Shape(intArrayOf(1,3)), floatArrayOf(0.4183f, 0.5209f, 0.0291f)))
    val sig_output = listOf(Tensor(Shape(intArrayOf(1,3)), floatArrayOf(0.4183f, 0.5209f, 0.0291f)))
    val fc2_output = listOf(Tensor(Shape(intArrayOf(1,2)), floatArrayOf(0.4183f, 0.5209f)))
    val softmax_output = listOf(Tensor(Shape(intArrayOf(1,2)), floatArrayOf(0.4183f, 0.5209f)))
    val label = listOf(Tensor(Shape(intArrayOf(1, 2)), floatArrayOf(0.7095f, 0.0942f)))
    val fc1 = FullyConnectedLayer(Shape(intArrayOf(1, 3)), Shape(intArrayOf(1, 3)))
    val sig = SigmoidActivation(Shape(intArrayOf(1, 3)))
    val fc2 = FullyConnectedLayer(Shape(intArrayOf(1, 3)), Shape(intArrayOf(1, 2)))
    val softmax = SoftmaxLayer(Shape(intArrayOf(1, 2)))
    val crossEntropyLoss = CrossEntropyLoss()

    @Test
    fun testMarkusNumbersFC(){
        fc1.setWeightsForTesting(fc1_bias, fc1_weights)
        fc2.setWeightsForTesting(fc2_bias, fc2_weights)
        fc1.forward(input, fc1_output)
        Assert.assertEquals(fc1_output[0].elements[0], -0.0469f, EPSILON)
        Assert.assertEquals(fc1_output[0].elements[1], 0.2406f, EPSILON)
        Assert.assertEquals(fc1_output[0].elements[2], 0.0561f, EPSILON)

        sig.forward(fc1_output, sig_output)
        Assert.assertEquals(sig_output[0].elements[0], 0.4883f, EPSILON)
        Assert.assertEquals(sig_output[0].elements[1], 0.5599f, EPSILON)
        Assert.assertEquals(sig_output[0].elements[2], 0.5140f, EPSILON)

        fc2.forward(sig_output, fc2_output)
        Assert.assertEquals(fc2_output[0].elements[0], -0.0728f, EPSILON)
        Assert.assertEquals(fc2_output[0].elements[1], 0.0229f, EPSILON)

        softmax.forward(fc2_output, softmax_output)
        Assert.assertEquals(softmax_output[0].elements[0], 0.4761f, EPSILON)
        Assert.assertEquals(softmax_output[0].elements[1], 0.5239f, EPSILON)

        crossEntropyLoss.differentiate(softmax_output, label)
        Assert.assertEquals(softmax_output[0].deltas[0], -1.4901f, EPSILON)
        Assert.assertEquals(softmax_output[0].deltas[1], -0.1798f, EPSILON)

        softmax.backward(softmax_output, fc2_output)
        Assert.assertEquals(fc2_output[0].deltas[0], -0.3268f, EPSILON)
        Assert.assertEquals(fc2_output[0].deltas[1], 0.3268f, EPSILON)

        fc2.backward(fc2_output, sig_output)
        Assert.assertEquals(sig_output[0].deltas[0], 0.1803f, EPSILON)
        Assert.assertEquals(sig_output[0].deltas[1], 0.2261f, EPSILON)
        Assert.assertEquals(sig_output[0].deltas[2], -0.3567f, EPSILON)

        sig.backward(sig_output, fc1_output)
        Assert.assertEquals(fc1_output[0].deltas[0], 0.0451f, EPSILON)
        Assert.assertEquals(fc1_output[0].deltas[1], 0.0557f, EPSILON)
        Assert.assertEquals(fc1_output[0].deltas[2], -0.0891f, EPSILON)

        fc1.calculateDeltaWeights(fc1_output, input)
        Assert.assertEquals(fc1.weightmatrix.deltas[0], 0.0188f, EPSILON)
        Assert.assertEquals(fc1.weightmatrix.deltas[1], 0.0235f, EPSILON)
        Assert.assertEquals(fc1.weightmatrix.deltas[2], 0.0013f, EPSILON)
        Assert.assertEquals(fc1.weightmatrix.deltas[3], 0.0233f, EPSILON)
        Assert.assertEquals(fc1.weightmatrix.deltas[4], 0.0290f, EPSILON)
        Assert.assertEquals(fc1.weightmatrix.deltas[5], 0.0016f, EPSILON)
        Assert.assertEquals(fc1.weightmatrix.deltas[6], -0.0373f, EPSILON)
        Assert.assertEquals(fc1.weightmatrix.deltas[7], -0.0464f, EPSILON)
        Assert.assertEquals(fc1.weightmatrix.deltas[8], -0.0026f, EPSILON)

        Assert.assertEquals(fc1.bias.deltas[0], 0.0451f, EPSILON)
        Assert.assertEquals(fc1.bias.deltas[1], 0.0557f, EPSILON)
        Assert.assertEquals(fc1.bias.deltas[2], -0.0891f, EPSILON)

        fc2.calculateDeltaWeights(fc2_output, sig_output)
        Assert.assertEquals(fc2.weightmatrix.deltas[0], -0.1596f, EPSILON)
        Assert.assertEquals(fc2.weightmatrix.deltas[1], -0.1830f, EPSILON)
        Assert.assertEquals(fc2.weightmatrix.deltas[2], -0.1680f, EPSILON)
        Assert.assertEquals(fc2.weightmatrix.deltas[3], 0.1596f, EPSILON)
        Assert.assertEquals(fc2.weightmatrix.deltas[4], 0.1830f, EPSILON)
        Assert.assertEquals(fc2.weightmatrix.deltas[5], 0.1680f, EPSILON)

        Assert.assertEquals(fc2.bias.deltas[0], -0.3268f, EPSILON)
        Assert.assertEquals(fc2.bias.deltas[1], 0.3268f, EPSILON)

        //fc1.backward(fc1_output, input)
    }


    @Test
    fun testMarkusNumbersConv2D(){
        // Init layer
        val conv2D_layer = Conv2DLayer( inputShape = Shape(intArrayOf(4,3,2)),
                outShape = Shape(intArrayOf(3,2,2)),
                filterShape = Shape(intArrayOf(2,2,2)),
                numOfFilters = 2)

        val conv2D_weights = floatArrayOf(  0.1f, -0.2f, 0.3f, 0.4f, 0.7f, 0.6f, 0.9f, -1.1f,
                                            0.37f, -0.9f, 0.32f, 0.17f, 0.9f, 0.3f, 0.2f, -0.7f)
        val conv2DLayer_bias = floatArrayOf(0.0f, 0.0f)
        conv2D_layer.setWeightsForTesting(Tensor(Shape(intArrayOf(2,2,2,2)), conv2D_weights),
                                          Tensor (Shape(intArrayOf(2)), conv2DLayer_bias ))

        // Init in and out tensor
        val out_tensors = listOf<Tensor>(Tensor(Shape(intArrayOf(3,2,2))))
        val in_tensors = listOf<Tensor>(
                Tensor(Shape(intArrayOf(4,3,2)),
                        floatArrayOf(0.1f, -0.2f, 0.5f, 0.6f, 1.2f, 1.4f, 1.6f, 2.2f, 0.01f, 0.2f, -0.3f, 4.0f, 0.9f,
                                0.3f, 0.5f, 0.65f, 1.1f, 0.7f, 2.2f, 4.4f, 3.2f, 1.7f, 6.3f, 8.2f)))

        // Forward Pass
        conv2D_layer.forward(in_tensors, out_tensors)
        val out_conv2D = out_tensors.get(0)

        //Set madeup deltas
        out_conv2D.deltas = floatArrayOf(0.1f, 0.33f, -0.6f, -0.25f, 1.3f, 0.01f, -0.5f, 0.2f, 0.1f, -0.8f, 0.81f, 1.1f)

        // Backward Pass
        conv2D_layer.backward(out_tensors, in_tensors)

        val modifiedKernelForBackward = conv2D_layer.rotatedTransposedKernel
        val deltas_backward = in_tensors.get(0).deltas
        val delta_weights = conv2D_layer.getKernel.deltas

        // Compare results with Markus' numbers
        //Forward
        //Assert.assertEquals(out_conv2D.elements[0], 2.0f, EPSILON)
        // Modified Kernel
        Assert.assertEquals(modifiedKernelForBackward.get(0,0,0,0), 0.4f, EPSILON)


    }
}