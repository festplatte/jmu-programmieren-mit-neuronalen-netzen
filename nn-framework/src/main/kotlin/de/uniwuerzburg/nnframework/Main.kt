package de.uniwuerzburg.nnframework

import java.io.File

fun main(args: Array<String>) {
    val mnistData = readFiles("/path/to/mnistfiles")
    val img = mnistData.keys.first()
    println(img)
    println(mnistData[img])

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