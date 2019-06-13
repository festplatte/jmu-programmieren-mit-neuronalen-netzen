package de.uniwuerzburg.nnframework

import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.loss.LossFkt

class SGDTrainer(private val batchSize: Int = 1,
                 private val learningRate: Float,
                 private val amountEpochs: Int,
                 private val lossFkt: LossFkt,
                 private val shuffle: Boolean = true,
                 private val updateMechanism: SGDFlavor) {
    /**
     * Bekommt die rohen Inputdaten zum Trainieren mit Labels und führt das gesamte Training durch.
     * Dabei wird für jeden Batch der Forward- und Backwardpass auf dem Netzwerk berechnet und
     * anschließend die Gewichte geupdatet anhand des updateMechanism.
     * @param network das zu trainierende Netzwerk
     * @param data die Trainigsdaten mit Zielwerten (Labels)
     */
    fun <T> optimize(network: Network<T>, data: LinkedHashMap<T, Tensor>) {
        var dataList = data.keys.toList()
        if (shuffle) dataList = dataList.shuffled()

        for (epoch in 0..amountEpochs) {
            println("Start epoch $epoch/$amountEpochs")

            var batchOffset = 0
            while (batchOffset < dataList.size) {
                val processedData = if (batchOffset + batchSize > dataList.size) dataList.size else batchOffset + batchSize
                val batchData = dataList.subList(batchOffset, processedData)
                val batchLabels = batchData.map { data[it] ?: throw IllegalArgumentException("no label for data") }

                val predictedLabels = network.forward(batchData)
                val loss = lossFkt.calculate(predictedLabels, batchLabels)
                network.backprop(loss)
                // TODO update weights

                println("Epoch: $epoch - Data: $processedData/${dataList.size} - Loss: $loss")

                batchOffset += batchSize
            }
        }

        println("Training finished")
    }

}