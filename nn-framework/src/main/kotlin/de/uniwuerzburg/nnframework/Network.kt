package de.uniwuerzburg.nnframework

import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.layers.FullyConnectedLayer
import de.uniwuerzburg.nnframework.layers.InputLayer
import de.uniwuerzburg.nnframework.layers.Layer
import de.uniwuerzburg.nnframework.layers.WeightLayer

/**
 * Netzwerk führt Forward-, Backward-Pass und Weightupdates auf den Layern aus.
 */
class Network<T>(private val input: InputLayer<T>,
                 private val layers: List<Layer>) {

    /** Größe des letzten verarbeiteten Batches */
    private var batchSize = 0

    /** Map mit Output-Tensorenliste in Batchgröße für jeden Layer */
    private val dataMap = LinkedHashMap<Layer, List<Tensor>>()


    fun updateWeights(updater: (value: Float, delta: Float) -> Float) {
        for (layer in layers) {
            if (layer is WeightLayer) {
                layer.updateWeights(updater)
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