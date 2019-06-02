package de.uniwuerzburg.nnframework

class SGDTrainer(private val batchSize: Int = 1,
                 private val learningRate: Float,
                 private val amountEpochs: Int,
                 private val shuffle: Boolean = true,
                 private val updateMechanism: SGDFlavor) {
    fun <T> optimize(network: Network<T>, data: List<T>) {

    }

}