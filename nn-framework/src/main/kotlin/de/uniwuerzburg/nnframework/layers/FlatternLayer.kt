package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Kopiert alle Elemente zwischen den Tensoren. Dabei sollten die Tensoren unterschiedliche Shapes haben. (ineffizient)
 */
class FlatternLayer(override val outputShape: Shape) : ActivationLayer {
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        inTensors.forEachIndexed { index, tensor ->
            System.arraycopy(tensor.elements, 0, outTensors[index].elements, 0, tensor.elements.size)
        }
    }

    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        outTensors.forEachIndexed { index, tensor ->
            System.arraycopy(tensor.elements, 0, inTensors[index].elements, 0, tensor.elements.size)
        }
    }
}