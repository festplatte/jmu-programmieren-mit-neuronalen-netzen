package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.data.mult

/**
 * @author vb
 */

/**
 * zu klären:
 * - warum inShape und outShape? die Tensoren haben bereits eine Shape. -> zur Initialisierung der Weightmatrix
 */
class FullyConnectedLayer(private val weightmatrix: Tensor,
                          private val bias: Tensor,
                          private val inShape: Shape,
                          private val outShape: Shape) : Layer {

    init {
        // Initialize weight matrix W and the bias

        // W: Fully connected, e.g. one weight between each pair contained in the in and the outShape
        // For each element in the inTensor there is a connection to each element of the outTensor
        for (i in 0 until weightmatrix.shape.dimensions) {
            //Init with values between -1 and 1
        }
        weightmatrix.elements.map { /* zufallszahl */ }

    }

    /*
    * The forward pass fills the elements of the outTensors,
    * while having access to the elements in inTensors:
    * inTensors * weightmatrix + bias = outTensors
    */
    override fun forward(inTensors: List<Tensor>, outTensors: List<Tensor>) {
        val inTensorsIterator = inTensors.iterator()
        for (inTensor in inTensorsIterator) {
            //Returns a tensor, however the existing outTensor should be filled ...
            mult(inTensor, weightmatrix) //.add(bias) TODO add bias after Michi has finished his Tensor class
        }

    }

    /*
    * The backward pass fills in the deltas of the inTensors,
    * while having access to the deltas in outTensors as well as to the elements of the inTensors
    */
    override fun backward(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        // TODO berechne die deltas der inTensors mit den deltas der outTensors und Daten beider Tensoren
    }

    /*
    * The calculateDeltaWeights function calculates the delta weights by
    * using the elements of the inTensors and the deltas of the outTensors
    */
    override fun calculateDeltaWeights(outTensors: List<Tensor>, inTensors: List<Tensor>) {
        // TODO berechne die Deltas der Gewichte mit Hilfe der Deltas der outTensors und Daten der inTensors
    }
}