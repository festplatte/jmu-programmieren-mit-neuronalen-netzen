package de.uniwuerzburg.nnframework

import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.layers.*
import kotlin.math.sqrt

/**
 * Netzwerk führt Forward-, Backward-Pass und Weightupdates auf den Layern aus.
 */
class Network<T>(private val input: InputLayer<T>,
                 private val layers: List<Layer>) {

    /** Größe des letzten verarbeiteten Batches */
    private var batchSize = 0

    /** Map mit Output-Tensorenliste in Batchgröße für jeden Layer */
    private val dataMap = LinkedHashMap<Layer, List<Tensor>>()


    fun updateWeights(updateMechanism: SGDFlavor, learningRate: Float) {
        if (updateMechanism == SGDFlavor.STOCHASTIC_GRADIENT_DESCENT) {
            for (layer in layers) {
                if (layer is FullyConnectedLayer) {
                    //Update Bias
                    for (i in layer.bias.deltas.indices) {
                        layer.bias.elements[i] -= learningRate * layer.bias.deltas[i]
                        layer.bias.deltas[i] = 0f
                    }
                    //Update weightmatrix
                    for (i in layer.weightmatrix.deltas.indices) {
                        layer.weightmatrix.elements[i] -= learningRate * layer.weightmatrix.deltas[i]
                        layer.weightmatrix.deltas[i] = 0f
                    }
                }
                if (layer is Conv2DLayer) {
                    //Update Bias
                    for (i in layer.bias.deltas.indices) {
                        layer.bias.elements[i] -= learningRate * layer.bias.deltas[i]
                        layer.bias.deltas[i] = 0f
                    }
                    //Update filters
                    for (i in layer.kernel.deltas.indices) {
                        layer.kernel.elements[i] -= learningRate * layer.kernel.deltas[i]
                        layer.kernel.deltas[i] = 0f
                    }
                }
            }
        } else if (updateMechanism == SGDFlavor.ADAM) {
            var m = 0f;
            var v = 0f;
            var beta1 = 0.9f
            var beta2 = 0.999f
            var epsilon = 0.00000001f
            for (layer in layers) {
                if (layer is FullyConnectedLayer) {
                    //Update Bias
                    for (i in layer.bias.deltas.indices) {
                        //m = beta1 * m + (1 - beta1) * layer.bias.deltas[i]
                        //v = beta2 * v + (1 - beta1) * layer.bias.deltas[i] * layer.bias.deltas[i]
                        layer.bias.elements[i] -= (learningRate * m) / (sqrt(v) * epsilon)
                        layer.bias.deltas[i] = 0f
                    }
                    //Update weightmatrix
                    for (i in layer.weightmatrix.deltas.indices) {
                        //m = beta1 * m + (1 - beta1) * layer.weightmatrix.deltas[i]
                        //v = beta2 * v + (1 - beta1) * layer.weightmatrix.deltas[i] * layer.weightmatrix.deltas[i]
                        layer.weightmatrix.elements[i] -= (learningRate * m) / (sqrt(v) * epsilon)
                        layer.weightmatrix.deltas[i] = 0f
                    }
                }
                if (layer is Conv2DLayer) {
                    //Update Bias
                    for (i in layer.bias.deltas.indices) {
                        //m = beta1 * m + (1 - beta1) * layer.bias.deltas[i]
                        //v = beta2 * v + (1 - beta1) * layer.bias.deltas[i] * layer.bias.deltas[i]
                        layer.bias.elements[i] -= (learningRate * m) / (sqrt(v) * epsilon)
                        layer.bias.deltas[i] = 0f
                    }
                    //Update filters
                    for (i in layer.kernel.deltas.indices) {
                        //m = beta1 * m + (1 - beta1) * layer.kernel.deltas[i]
                        //v = beta2 * v + (1 - beta1) * layer.kernel.deltas[i] * layer.kernel.deltas[i]
                        layer.kernel.elements[i] -= (learningRate * m) / (sqrt(v) * epsilon)
                        layer.kernel.deltas[i] = 0f
                    }
                }
            }
        }
    }

    /**
     * Führt den Forward-Pass durch und gibt die Ausgabe des letzten Layers zurück.
     * @param rawDataList Batch der Input-Daten zur Verarbeitung mit InputLayer
     * @return Ausgabe-Tensoren der letzten Schicht
     */
    fun forward(rawDataList: List<T>): List<Tensor> {
        initDataMap(rawDataList.size)

        var data = input.forward(rawDataList)
        layers.forEach { layer ->
            val outputData = dataMap[layer] ?: throw IllegalArgumentException("no outputData for layer")
            layer.forward(data, outputData)
            data = outputData
        }
        return data
    }

    /**
     * Initialisiert die leeren Output-Tensoren für jeden Layer wenn sich die Batchsize geändert hat.
     * @param batchSize Anzahl der Output-Tensoren pro Layer
     */
    private fun initDataMap(batchSize: Int) {
        if (batchSize != this.batchSize) {
            this.batchSize = batchSize
            dataMap.clear()

            layers.forEach {
                val data = List(batchSize) { _ -> Tensor(it.outputShape) }
                dataMap[it] = data
            }
        }
    }

    /**
     * Führt den Backwardpass durch.
     * @param loss Der berechnete Loss für den Output des Forwardpass
     */
    fun backprop(lastOutput: List<Tensor>) {
        for (i in (1..layers.lastIndex).reversed()) {
            val layer = layers[i]
            val prevLayer = layers[i - 1]
            val outputData = dataMap[layer] ?: throw IllegalArgumentException("no outputData for layer")
            val inputData = dataMap[prevLayer] ?: throw IllegalArgumentException("no outputData for layer")

            layer.backward(outputData, inputData)
            if (layer is WeightLayer) {
                layer.calculateDeltaWeights(outputData, inputData)
            }
        }
    }

}