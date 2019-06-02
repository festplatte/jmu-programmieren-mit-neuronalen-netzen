package de.uniwuerzburg.nnframework

class SGDTrainer(private val batchSize: Int = 1,
                 private val learningRate: Float,
                 private val amountEpochs: Int,
                 private val shuffle: Boolean = true,
                 private val updateMechanism: SGDFlavor) {
    /**
     * Bekommt die rohen Inputdaten zum Trainieren mit Labels und führt das gesamte Training durch.
     * Dabei wird für jeden Batch der Forward- und Backwardpass auf dem Netzwerk berechnet und
     * anschließend die Gewichte geupdatet anhand des updateMechanism.
     */
    fun <T> optimize(network: Network<T>, data: List<T>) {

    }

}