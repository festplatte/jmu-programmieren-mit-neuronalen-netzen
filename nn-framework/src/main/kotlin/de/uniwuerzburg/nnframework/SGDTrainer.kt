package de.uniwuerzburg.nnframework

import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.loss.LossFkt
import kotlin.math.exp

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
    fun <T> optimize(network: Network<T>, data: Map<T, Tensor>) {
        var dataList = data.keys.toList()
        if (shuffle) dataList = dataList.shuffled()

        for (epoch in 0..amountEpochs) {
            println("Start epoch $epoch/$amountEpochs")

            var batchOffset = 0
            while (batchOffset < dataList.size) {
                val processedData = if (batchOffset + batchSize > dataList.size) dataList.size else batchOffset + batchSize
                val batchData = dataList.subList(batchOffset, processedData)
                val batchLabels = batchData.map { data[it] ?: throw IllegalArgumentException("no label for data") }

                val forwardOutput = network.forward(batchData)
                val loss = lossFkt.calculate(forwardOutput, batchLabels)
                val accuracy = calcAccuracy(forwardOutput, batchLabels)
                lossFkt.differentiate(forwardOutput, batchLabels)
                network.backprop(forwardOutput)
                // TODO update weights
                network.updateWeights(updateMechanism, learningRate)

                println("Epoch: $epoch - Data: $processedData/${dataList.size} - Loss: $loss - Accuracy: $accuracy")

                batchOffset += batchSize
            }
        }

        println("Training finished")
    }

    /**
     * Berechnet die Accuracy anhand der gegebenen und erwarteten Tensoren.
     */
    private fun calcAccuracy(actual: List<Tensor>, expected: List<Tensor>): Float {
        var correctlyPredicted = 0
        for (i in 0 until actual.size) {
            if (getHighestIndex(actual[i]) == getHighestIndex(expected[i])) {
                correctlyPredicted++
            }
        }
        return correctlyPredicted.toFloat() / actual.size.toFloat()
    }

    /**
     * Gibt den Index des größten Tensor-Elements zurück.
     */
    private fun getHighestIndex(tensor: Tensor): Int {
        var highest: Float = 0f
        var highestIndex: Int = 0
        tensor.elements.forEachIndexed { index, fl ->
            if (fl > highest) {
                highest = fl
                highestIndex = index
            }
        }
        return highestIndex
    }
}