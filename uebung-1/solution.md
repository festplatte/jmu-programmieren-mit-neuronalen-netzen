# Lösung zu Übung 1

## Aufgabe 2 (`mnist-tutorial.py`)

### b)

```python
10000/10000 [==============================] - 0s 25us/sample - loss: 0.0684 - acc: 0.9802 # evaluation auf Testdaten
60000/60000 [==============================] - 1s 18us/sample - loss: 0.0218 - acc: 0.9932 # Evaluation auf Trainingsdaten
```

Accuracy ist höher für Trainingsdaten, da Model explizit darauf trainiert wurde.

### c)

ohne zusätzlichem Layer siehe b)

mit zusätzlichem Layer

```py
10000/10000 [==============================] - 0s 31us/sample - loss: 0.0838 - acc: 0.9787 # evaluation auf Testdaten
60000/60000 [==============================] - 2s 25us/sample - loss: 0.0265 - acc: 0.9916 # Evaluation auf Trainingsdaten
```

Die Accuracy sinkt für beide Evaluationen, aber geringe Abweichung.

### d)

mit nur einer Zelle

```py
10000/10000 [==============================] - 0s 15us/sample - loss: 1.6894 - acc: 0.3269
60000/60000 [==============================] - 1s 10us/sample - loss: 1.6750 - acc: 0.3275
```

Die Accuracy sinkt drastisch.

### e)

```py
10000/10000 [==============================] - 0s 25us/sample - loss: 0.0786 - acc: 0.9760
60000/60000 [==============================] - 1s 19us/sample - loss: 0.0361 - acc: 0.9884
```

etwas niedrigere Accuracy.

### f)

Evaluationsergebnis für gespeichertes Model `mnist-tutorial.h5`

```py
10000/10000 [==============================] - 0s 24us/sample - loss: 0.0717 - acc: 0.9794
60000/60000 [==============================] - 1s 18us/sample - loss: 0.0241 - acc: 0.9922
```

## Aufgabe 3 (`mnist-load-model.py`)

### a)

gleiches Ergebnis bei Evaluation auf Testdaten wie 2f)

```py
10000/10000 [==============================] - 0s 24us/sample - loss: 0.0717 - acc: 0.9794
```

### b)

Das sollte mit der `predict` Methode funktionieren (siehe https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#predict).

### c)

Das manuelle Ausrechnen ergibt auch eine Accuracy von `0.9794`
