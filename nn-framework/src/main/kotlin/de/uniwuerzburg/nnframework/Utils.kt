package de.uniwuerzburg.nnframework

/*
 * @author vb
* */

/**
 * Die built-in map()-Funktion veraendert die Elemente des ursp. Array nicht in place.
 * Stattdessen gibt sie eine Liste mit dem Ergebnis, das durch Transformation jedes Elements entsteht, zurueck.
 * Will man das nicht, muss man diese Funktion nutzen, um das Array in place zu modifizieren
 */
fun FloatArray.mapInPlace(transform: (Float) -> Float) {
    for (i in this.indices) {
        this[i] = transform(this[i])
    }
}
