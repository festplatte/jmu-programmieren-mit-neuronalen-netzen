package de.uniwuerzburg.nnframework

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor
import de.uniwuerzburg.nnframework.data.initializeWeights
import de.uniwuerzburg.nnframework.data.printTensor
import de.uniwuerzburg.nnframework.layers.FullyConnectedLayer
import java.io.File

fun main(args: Array<String>) {
    /*
    val mnistData = readFiles("/path/to/mnistfiles")
    val img = mnistData.keys.first()
    println(img)
    println(mnistData[img])
    */
    /*
    val test = Tensor(Shape(intArrayOf(3)), FloatArray(3))
    initializeWeights(test)
    printTensor(test)

    val test2 = Tensor(Shape(intArrayOf(3,2)), FloatArray(6))
    initializeWeights(test2)
    printTensor(test2)
    */

}

fun readFiles(path: String): Map<String, Int> {
    val dir = File(path)
    val result = LinkedHashMap<String, Int>()
    if (dir.isDirectory) {
        dir.listFiles().filter { !it.isDirectory && it.absolutePath.endsWith(".image") }.forEach { imageFile ->
            val labelFile = File(imageFile.absolutePath.replace(".image", ".label"))
            if (labelFile.exists()) {
                val image = imageFile.readText()
                val label = labelFile.readText().toInt()
                result.put(image, label)
            }
        }
    }
    return result
}