# Resolución de problemas basados en búsquedas

## Tarea 4. Movimiento del caballo

En esta tarea se adjunta un código de ejemplo para ayudar a la realización de la misma.

Esta tarea consiste en probar el algoritmo A\* frente a la búsqueda primero en anchura con grafo. Para ello se va a implementar un tablero de ajedrez en el cual se indicará la posición de inicio de una ficha del caballo y su posición final, para que el algoritmo de búsqueda indique los movimientos que tiene que realizar el caballo.

En esta tarea se pide que se implementen dos heurísticas para usar con el algoritmo A\*.

Entregables:

El código con la implementación
Un documento pdf donde se muestre la comparación entra la búsqueda en anchura y las dos heurísticas implementadas. Para ello es necesario que exista una tabla en la que se indique diferentes profundidades a las que se encuentra la solución y para ellas los nodos expandidos y el tiempo de ejecución.
Además, en el documento pdf, es necesario que se expliquen en que consisten las heurísticas implementadas.

## Solución

Primero de todo se realizará una adaptación del código para el funcionamiento del código facilitado para los movimientos del caballo. Hemos indicado como inicial que el caballo se encuentra en la casilla A1 (0,0) y debe de ir a la casilla H8 (7,7). En la ejecución del código, están comentados todos las ejecuciones excepto la primera, y el resto comentada, pero en la entrega del mismo se irán añadiendo todas las salidas según el método de búsqueda a utilizar. Para que el consumo de recursos sea equitativo, se procederá al borrado de los outputs correspondientes, por otro lado se tendrá un control del consumo de recursos del sistema en caso de que el se consuma excesiva memoria. Las funciones heurísticas a utilizar serán la heurística Manhattan y la heurística Euclídea

```python
import os
import time
import heapq
import psutil
import functools

from collections import deque
from typing import Optional


def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)


def humanbytes(B: int) -> str:
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Statistics(object):
    amount = 0

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Statistics, cls).__new__(cls)
        return cls.instance

    def reset(self):
        self.amount = 0

    def increase(self):
        self.amount += 1

    def get_amount(self):
        return self.amount

# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        Statistics().increase()
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


# ______________________________________________________________________________
# Queues: Stack, FIFOQueue, PriorityQueue
# Stack and FIFOQueue are implemented as list and collection.deque
# PriorityQueue is implemented here


class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

# ______________________________________________________________________________
# Uninformed Search algorithms


def breadth_first_tree_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = deque([Node(problem.initial)])  # FIFO queue

    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_tree_search(problem):
    """
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def depth_first_graph_search(problem):
    """
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None


def breadth_first_graph_search(problem):
    """
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

# ______________________________________________________________________________


class Invented(Problem):
    """
    Example of implementation of the problem.
    The states are three digits (from 100 to 999)
    The actions is to add or subtract some digit (+1, -1, +10, -10, +100, -100)
    """

    def __init__(self, initial=(0,0), goal=(7, 7)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = []
        moves = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                 (-2, -1), (-1, -2), (1, -2), (2, -1)]

        for move in moves:
            x = state[0] + move[0]
            y = state[1] + move[1]
            if 0 <= x < 8 and 0 <= y < 8:
                possible_actions.append(move)
        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        x = state[0] + action[0]
        y = state[1] + action[1]
        return x, y


    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        state = str(node.state)
        goal = str(self.goal)

        tmp = 0
        for cnt in range(3):
            tmp += abs(int(state[cnt]) - int(goal[cnt]))

        return tmp

    # Ahora aplicaremos funcion Euristica Manhattan
    def manhattan(self, node):
        deltaX = abs(node.state[0] - self.goal[0])
        deltaY = abs(node.state[1] - self.goal[1])
        return deltaX + deltaY

    # Ahora aplicaremos funcion Euristica Euclidea
    def euclidea(self, node):
        deltaX = abs(node.state[0] - self.goal[0])
        deltaY = abs(node.state[1] - self.goal[1])
        return (deltaX ** 2 + deltaY ** 2) ** 0.5

if __name__ == '__main__':
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss


    problem: Problem = Invented()

    start = time.process_time()
    # Refers to the ime the CPU was busy processing the program’s instructions.
    # The time spent waiting for other task to complete (like I/O operations) is not included in the CPU time.
    solution: Optional[Node] = breadth_first_tree_search(problem)
    #solution: Optional[Node] = breadth_first_graph_search(problem)
    #solution: Optional[Node] = astar_search(problem, problem.manhattan)
    #solution: Optional[Node] = astar_search(problem, problem.euclidea)

    elapsed = time.process_time() - start

    if solution is not None:
        print("Nodos expandidos: ", Statistics().get_amount())
        print("Profundidad de la solución: ", solution.depth)
        print("Nodos:", solution.path(), sep='\n\t')
        print("Acciones:", solution.solution(), sep='\n\t')

    # print('\nMemory usage finally: %s (%.2f%%)\n' % (humanbytes(process.memory_info().rss), process.memory_percent()))
    print('\nMemory usage: %s\n' % humanbytes(process.memory_info().rss - mem_before))
    print('CPU Execution time: %.6f seconds' % elapsed)



```

```


#####################################################
#       Búsqueda en profundidad usando árbol        #
#####################################################

    Nodos expandidos:  2838

    Profundidad de la solución:  6

    Nodos:
    	[<Node (0, 0)>, <Node (2, 1)>, <Node (4, 2)>, <Node (6, 3)>, <Node (7, 5)>, <Node (5, 6)>, <Node (7, 7)>]

    Acciones:
    	[(2, 1), (2, 1), (2, 1), (1, 2), (-2, 1), (2, 1)]

    Memory usage: 768.00 KB

    CPU Execution time: 0.037188 seconds

#####################################################
#       Búsqueda en profundidad usando grafo        #
#####################################################

Nodos expandidos:  55

Profundidad de la solución:  6

Nodos:
	[<Node (0, 0)>, <Node (2, 1)>, <Node (4, 2)>, <Node (6, 3)>, <Node (7, 5)>, <Node (5, 6)>, <Node (7, 7)>]

Acciones:
	[(2, 1), (2, 1), (2, 1), (1, 2), (-2, 1), (2, 1)]

Memory usage: 0.0 Byte

CPU Execution time: 0.001044 seconds


#####################################################
#              Heurística Manhattan                  #
#####################################################

Nodos expandidos:  13

Profundidad de la solución:  6

Nodos:
	[<Node (0, 0)>, <Node (1, 2)>, <Node (2, 4)>, <Node (4, 5)>, <Node (3, 7)>, <Node (5, 6)>, <Node (7, 7)>]

Acciones:
	[(1, 2), (1, 2), (2, 1), (-1, 2), (2, -1), (2, 1)]

Memory usage: 0.0 Byte

CPU Execution time: 0.000682 seconds

#####################################################
#              Heurística Euclídea                  #
#####################################################

Nodos expandidos:  13

Profundidad de la solución:  6

Nodos:
	[<Node (0, 0)>, <Node (1, 2)>, <Node (3, 3)>, <Node (5, 4)>, <Node (7, 5)>, <Node (5, 6)>, <Node (7, 7)>]

Acciones:
	[(1, 2), (2, 1), (2, 1), (2, 1), (-2, 1), (2, 1)]

Memory usage: 0.0 Byte

CPU Execution time: 0.000613 seconds


```

---

---

---

---

---

---

---

---

---

---

---

## Exposición de los datos obtenidos.

Se procede a realizar una tabla comparativa para ver cual es el mejor modelo de trabajo para este caso.
Se descarta directamente `Búsqueda en profundidad usando grafo` debido a que este proceso ha entrado en **bucle infinito** y por tanto su funcionalidad para este caso es nula.

|                            | **Búsqueda en anchura usando árbol** | **Búsqueda en anchura usando grafo** | <span style="color:green">**Heurística Manhattan**</span> | <span style="color:green">**Heurística Euclídea**</span> |
| -------------------------- | ------------------------------------ | ------------------------------------ | --------------------------------------------------------- | -------------------------------------------------------- |
| Nodos expandidos           | 2838                                 | 55                                   | **13**                                                    | **13**                                                   |
| Profundidad de la solución | **6**                                | **6**                                | **6**                                                     | **6**                                                    |
| Memoria usada finalmente   | 768.00 KB                            | **0.0 Byte**                         | **0.0 Byte**                                              | **0.0 Byte**                                             |
| Tiempo de ejecución de CPU | 0.037188 seconds                     | 0.001044 seconds                     | 0.000682 seconds                                          | **0.000613 seconds**                                     |

Cabe destacar de los datos obtenidos las siguientes reflexiones:
La cantidad de nodos expandidos, en este caso va a determinar un factor excluyente para los dos métodos de búsqueda en anchura ya que es muy superior la expansion de estos comparados con los métodos Heurísticos. Quedando solamente las búsquedas Heurísticas, solo destacar que tienen unis datos muy similares incluso en tiempo, puede deberse en este caso a que el valor de la distancia es muy corta en este caso, y debería de probarse cual es mas rápido por ejemplo en otros ejercicios como el mapa de carreteras de Rumanía.

---

## Conclusión

Como conclusión al estudio de datos obtenidos y las reflexiones realizadas, se puede determinar que los método de **Heurística Manhattan** y **Heurística Euclídea** son los mas eficientes de los 4 que se han realizado para el caso del ajedrez.

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

---

## Heurística Manhattan.

El método de búsqueda heurística Manhattan, también conocido como búsqueda por distancia de Manhattan, es una técnica utilizada en algoritmos de búsqueda que se basa en una heurística que calcula la distancia en términos de cuadras o manzanas entre dos puntos en una cuadrícula.

En este método, cada nodo en un grafo está representado por una ubicación en una cuadrícula. La heurística Manhattan calcula la distancia entre dos nodos, sumando las distancias de los componentes horizontal y vertical entre las dos ubicaciones. En otras palabras, la distancia Manhattan es la suma de las distancias de la fila y la columna entre dos nodos.

Por ejemplo, si un nodo está en la ubicación (2,3) y otro nodo está en la ubicación (5,7), la distancia Manhattan entre ellos sería |5-2| + |7-3| = 3 + 4 = 7.

La búsqueda heurística Manhattan utiliza esta heurística para guiar la búsqueda de una solución óptima. En lugar de explorar todos los nodos posibles en un grafo, el algoritmo se enfoca en los nodos que se espera que sean más prometedores en términos de distancia Manhattan a la solución deseada.

Este método es especialmente útil en problemas de búsqueda en laberintos o mapas, donde la distancia Manhattan es una buena medida de la distancia real que se necesita para llegar a un destino. El algoritmo de búsqueda heurística Manhattan puede ser más rápido y eficiente que otros métodos de búsqueda exhaustiva, como la búsqueda en profundidad o la búsqueda en anchura, porque se enfoca en nodos prometedores y evita explorar caminos poco prometedores.

## Heurística Euclidea.

El método de búsqueda heurística Euclídea, también conocido como búsqueda por distancia Euclídea, es una técnica utilizada en algoritmos de búsqueda que se basa en una heurística que calcula la distancia euclidiana entre dos puntos en un espacio n-dimensional.

En este método, cada nodo en un grafo está representado por una ubicación en un espacio n-dimensional. La heurística Euclídea calcula la distancia entre dos nodos utilizando la fórmula de la distancia euclidiana, que es la raíz cuadrada de la suma de los cuadrados de las diferencias de las componentes entre las dos ubicaciones. En otras palabras, la distancia euclidiana es la distancia en línea recta entre dos nodos.

Por ejemplo, si un nodo está en la ubicación (2,3) y otro nodo está en la ubicación (5,7), la distancia euclidiana entre ellos sería sqrt((5-2)^2 + (7-3)^2) = sqrt(9+16) = sqrt(25) = 5.

La búsqueda heurística Euclídea utiliza esta heurística para guiar la búsqueda de una solución óptima. En lugar de explorar todos los nodos posibles en un grafo, el algoritmo se enfoca en los nodos que se espera que sean más prometedores en términos de distancia Euclídea a la solución deseada.

Este método es especialmente útil en problemas de búsqueda en espacios n-dimensionales, donde la distancia Euclídea es una buena medida de la distancia real que se necesita para llegar a un destino. El algoritmo de búsqueda heurística Euclídea puede ser más rápido y eficiente que otros métodos de búsqueda exhaustiva, como la búsqueda en profundidad o la búsqueda en anchura, porque se enfoca en nodos prometedores y evita explorar caminos poco prometedores.

---

---

---

---

---

---



## Conclusiones de ambos métodos.

La gran diferencia entre los métodos de búsqueda heurística Manhattan y Euclídea es la forma en que calculan la distancia entre dos nodos en un grafo.

Mientras que la búsqueda heurística Manhattan utiliza la distancia de Manhattan, que se basa en la suma de las distancias horizontales y verticales entre dos nodos, la búsqueda heurística Euclídea utiliza la distancia euclidiana, que es la distancia en línea recta entre dos nodos.

La elección del método de búsqueda heurística dependerá del problema específico que se está abordando y de la naturaleza de los nodos en el grafo. En general, si los nodos en el grafo están ubicados en una cuadrícula, como en un laberinto o un mapa de la ciudad, la búsqueda heurística Manhattan puede ser más apropiada porque la distancia de Manhattan es una buena medida de la distancia real que se necesita para llegar a un destino. Si los nodos en el grafo están ubicados en un espacio continuo, la búsqueda heurística Euclídea puede ser más apropiada porque la distancia euclidiana es una buena medida de la distancia real entre dos puntos en el espacio.

En resumen, la diferencia clave entre los métodos de búsqueda heurística Manhattan y Euclídea es la forma en que se calcula la distancia entre dos nodos en un grafo, y la elección del método dependerá de la naturaleza del problema y de los nodos en el grafo.
