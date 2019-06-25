package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Kopiert alle Elemente zwischen den Tensoren. Dabei sollten die Tensoren unterschiedliche Shapes haben. (ineffizient)
 */
class FlatternLayer(override val outputShape: Shape) : Layer {
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        inTensors.forEachIndexed { index, tensor ->
            outTensors[index].elements = tensor.elements
        }
    }

    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        outTensors.forEachIndexed { index, tensor ->
            inTensors[index].deltas = tensor.deltas
        }
    }
}