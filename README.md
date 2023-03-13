# Resolución de problemas basados en búsquedas

## Tarea1: Experimentación de búsqueda en anchura y profundidad

El objetivo de está tarea es el de entender el código del programa adjunto y poder experimentar la búsqueda en anchura y profundidad.

Es necesario que se ejecute el código y se tomen los datos de tiempo de ejecución, memoria consumida y cantidad de nodos expandidos para:

- Búsqueda en anchura usando árbol.
- Búsqueda en anchura usando grafo.
- Búsqueda en profundidad usando grafo.
- Búsqueda en profundidad usando árbol.

## Tarea 2: Números

En esta tarea es necesario que utilices el código de la tarea 1. Crea un problema cuyos estados sean números de tres cifras y las posibles acciones sean sumar o restar una unidad a alguna de las cifras (+1, +10, +100, -1, -10, -100).

El problema en concreto tendrá una serie de números tabú que deberían de ser imposibles de alcanzar.

Es decir, la definición del problema es la siguiente:

Estado inicial: 789
Estado Objetivo: 269
Números tabú: 244, 253, 254, 343, 344, 353, 778, 779, 679, 689
Acciones: +1, +10, +100, -1, -10, -100

## Tarea 3: 8 puzzle

En esta tarea es necesario que utilices el código de la tarea 1 para implementar el problema del 8-puzzle.

El objetivo de esta tarea es que compruebes como se comporta el algoritmo de búsqueda en anchura en cuanto a la profundidad. Para ello es necesario que definas varios estados iniciales cuya solución se encuentre a distintas profundidades.

Entregables:

- El código con la implementación
- Un documento pdf donde se incluya:
- Estados iniciales que se encuentran a una profundidad de 1, 2, 3, 4 y 5 del nodo objetivo.
- Se tomen las métricas de tiempos de consumo y de memoria para la búsqueda de primero en anchura para los estados iniciales definidos utilizando árboles.
- Lo mismo que el punto anterior pero utilizando grafos en lugar de árboles.

## Tarea 4. Movimiento del caballo

En esta tarea se adjunta un código de ejemplo para ayudar a la realización de la misma.

Esta tarea consiste en probar el algoritmo A\* frente a la búsqueda primero en anchura con grafo. Para ello se va a implementar un tablero de ajedrez en el cual se indicará la posición de inicio de una ficha del caballo y su posición final, para que el algoritmo de búsqueda indique los movimientos que tiene que realizar el caballo.

En esta tarea se pide que se implementen dos heurísticas para usar con el algoritmo A\*.
s