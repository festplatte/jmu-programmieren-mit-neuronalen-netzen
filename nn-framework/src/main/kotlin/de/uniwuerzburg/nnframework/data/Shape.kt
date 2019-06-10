package de.uniwuerzburg.nnframework.data

/**
 * Beschreibt die Form eines Tensors.
 */
class Shape(val axis: IntArray) {
    /**
     * Berechnet das Volumen der Shape (multipliziert die Größe aller Achsen).
     */
    val volume: Int get() = axis.reduce { acc, i -> acc * i }

    /**
     * Gibt die Anzahl der Achsen/Dimensionen zurück.
     */
    val dimensions: Int get() = axis.size

    /**
     * Gibt die Größe einer Achse zurück.
     *
     * @param i Nummer der Achse
     */
    fun get(i: Int) = if (i < axis.size ) axis[i] else 1
}