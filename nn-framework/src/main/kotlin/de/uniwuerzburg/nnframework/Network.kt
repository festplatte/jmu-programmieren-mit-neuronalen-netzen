package de.uniwuerzburg.nnframework

import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.layers.InputLayer
import de.uniwuerzburg.nnframework.layers.Layer

/**
 * Netzwerk führt Forward-, Backward-Pass und Weightupdates auf den Layern aus.
 */
class Network<T>(private val input: InputLayer<T>,
                 private val layers: List<Layer>) {

    /** Größe des letzten verarbeiteten Batches */
    private var batchSize = 0

    /** Map mit Output-Tensorenliste in Batchgröße für jeden Layer */
    private val dataMap = LinkedHashMap<Layer, List<Tensor>>()


    // TODO methode für Weightupdates

    /**
     * Führt den Forward-Pass durch und gibt die Ausgabe des letzten Layers zurück.
     * @param rawDataList Batch der Input-Daten zur Verarbeitung mit InputLayer
     * @return Ausgabe-Tensoren der letzten Schicht
     */
    fun forward(rawDataList: List<T>): List<Tensor> {
        if (batchSize != rawDataList.size) {
            initDataMap(rawDataList.size)
        }

        var data = input.forward(rawDataList)
        dataMap.forEach { (layer, outputData) ->
            layer.forward(data, outputData)
            data = outputData
        }
        return data
    }

    /**
     * Initialisiert die leeren Output-Tensoren für jeden Layer.
     * @param batchSize Anzahl der Output-Tensoren pro Layer
     */
    private fun initDataMap(batchSize: Int) {
        this.batchSize = batchSize
        dataMap.clear()

        layers.forEach {
            val data = List(batchSize) { _ -> Tensor(it.outputShape) }
            dataMap[it] = data
        }
    }

    /**
     * Führt den Backwardpass durch.
     * @param loss Der berechnete Loss für den Output des Forwardpass
     */
    fun backprop(loss: List<Tensor>) {
        // TODO backward jedes Layers, calculateDeltaWeights der Layer
    }

}