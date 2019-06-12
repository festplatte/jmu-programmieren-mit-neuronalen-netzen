package de.uniwuerzburg.nnframework.layers

import de.uniwuerzburg.nnframework.data.Tensor

/**
 * InputLayer, der aus beliebigen Daten Tensoren produziert. Für jeden zu verwendenden Datentyp
 * muss eine eigene Implementierung vorliegen.
 */
interface InputLayer<T> {
    fun forward(rawDataList: List<T>): List<Tensor>
}