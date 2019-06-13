package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Interface, dass implementiert werden sollte, um eine Aktivierungsfunktion (z.B. ReLU, Sigmoid, TanH) zu erstellen.
 */
interface ActivationLayer: Layer {
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>)
    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>)
}