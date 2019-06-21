package de.uniwuerzburg.nnframework

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.layers.*
import de.uniwuerzburg.nnframework.loss.CrossEntropyLoss
import java.io.File

fun main(args: Array<String>) {
    //val mnistData = readFiles("/Users/michaelgabler/Repositories/jmu-machine-learning-for-nlp/exercise-3/MNIST PyTorch/data/test")
    //val mnistData = readFiles("C:/Users/simon/Documents/Master/MnistData/MNIST-Data/test")
    val mnistTrain = readFiles("C:/Users/Simon Englert/Documents/Studium/ML for NLP/MNIST PyTorch/data/train")
    val mnistTest = readFiles("C:/Users/Simon Englert/Documents/Studium/ML for NLP/MNIST PyTorch/data/test")

    val network = Network(ImageStringInputLayer(), listOf(
            FlatternLayer(Shape(intArrayOf(1, 784))),
            FullyConnectedLayer(Shape(intArrayOf(1, 784)), Shape(intArrayOf(1, 512))),
            SigmoidActivation(Shape(intArrayOf(1, 512))),
            FullyConnectedLayer(Shape(intArrayOf(1, 512)), Shape(intArrayOf(1, 10))),
            SoftmaxLayer(Shape(intArrayOf(1, 10)))
    ))
    val trainer = SGDTrainer(256, 0.001f, 10, CrossEntropyLoss(), true, SGDFlavor.STOCHASTIC_GRADIENT_DESCENT)
    trainer.optimize(network, mnistTrain)
    trainer.validate(network, mnistTest)
}

fun readFiles(path: String): Map<String, Tensor> {
    val dir = File(path)
    val result = LinkedHashMap<String, Tensor>()
    if (dir.isDirectory) {
        dir.listFiles().filter { !it.isDirectory && it.absolutePath.endsWith(".image") }.forEach { imageFile ->
            val labelFile = File(imageFile.absolutePath.replace(".image", ".label"))
            if (labelFile.exists()) {
                val image = imageFile.readText()
                val label = labelFile.readText().toInt()
                result.put(image, mapLabelToTensor(label, 10))
            }
        }
    }
    return result
}

fun mapLabelToTensor(label: Int, amountLabels: Int): Tensor {
    val elements = FloatArray(amountLabels)
    for (i in 0 until elements.size) {
        elements[i] = if (label == i + 1) 1f else 0f
    }
    return Tensor(Shape(intArrayOf(1, amountLabels)), elements)

}
