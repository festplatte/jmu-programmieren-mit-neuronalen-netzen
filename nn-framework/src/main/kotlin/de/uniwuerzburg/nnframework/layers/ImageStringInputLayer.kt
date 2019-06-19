package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Shape
import de.uniwuerzburg.nnframework.data.Tensor

/**
 * Transformiert Strings mit Helligkeitwerten in Tensoren.
 */
class ImageStringInputLayer: InputLayer<String> {
    private val ROW_SPLIT = 10.toChar().toString()
    //private val ROW_SPLIT = System.lineSeparator()
    private val COLUMN_SPLIT = " "

    override fun forward(rawDataList: List<String>): List<Tensor> {
        return rawDataList.map { data ->
            val valuesList = data.split(ROW_SPLIT).map { row -> row.split(COLUMN_SPLIT).map { it.toFloat() } }
            val shape = Shape(intArrayOf(valuesList.size, valuesList[0].size))
            val valuesArray = FloatArray(shape.volume)
            var index = 0
            valuesList.forEach { row ->
                System.arraycopy(row.toFloatArray(), 0, valuesArray, index, row.size)
                index += row.size
            }
            Tensor(shape, valuesArray)
        }
    }
}