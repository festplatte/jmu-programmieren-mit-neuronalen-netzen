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

    /**
     * This method can be used to check if two shapes match each other
     */
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Shape

        if (!axis.contentEquals(other.axis)) return false

        return true
    }

    /**
     *  Overrides the hashCode function based on the axis-IntArray
     */
    override fun hashCode(): Int {
        return axis.contentHashCode()
    }


}