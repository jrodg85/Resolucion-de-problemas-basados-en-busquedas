# Resolución de problemas basados en búsquedas

## Tarea 3: 8 puzzle

En esta tarea es necesario que utilices el código de la tarea 1 para implementar el problema del 8-puzzle.

El objetivo de esta tarea es que compruebes como se comporta el algoritmo de búsqueda en anchura en cuanto a la profundidad. Para ello es necesario que definas varios estados iniciales cuya solución se encuentre a distintas profundidades.

Entregables:

- El código con la implementación
- Un documento pdf donde se incluya:
- Estados iniciales que se encuentran a una profundidad de 1, 2, 3, 4 y 5 del nodo objetivo.
- Se tomen las métricas de tiempos de consumo y de memoria para la búsqueda de primero en anchura para los estados iniciales definidos utilizando árboles.
- Lo mismo que el punto anterior pero utilizando grafos en lugar de árboles.

## Solución

La tarea a ejecutar es dado un inicial aleatoria, la posición final mediante intercambio de números entre si por parte del 0 tiene que ser (1,2,3,4,5,6,7,8,0).

Tal y como pide el enunciado, se realizará con estados iniciales en los que la profundidad sea 1, 2, 3, 4 y 5 con el nodo objetivo.

Ya que puede haber combinaciones por del 8 puzzle que no tengan solución, para tener una profundidad idónea, se procede a usar el siguiente repositorio de github (https://github.com/NiharG15/8-Puzzle) para poder realizar ejercicios de profundidad 1 2 3 4 y 5.

- Profundidad 1: (1,2,3,4,5,0,7,8,6)
- Profundidad 2: (1,2,3,4,0,6,7,5,8)
- Profundidad 3: (1,2,3,0,5,6,4,7,8)
- Profundidad 4: (1,3,0,4,2,5,7,8,6)
- Profundidad 5: (2,0,3,1,4,6,7,5,8)

En el caso que nos trata, deberemos de realizar la misma acción en las 5 profundidades los cuatro métodos.
Para no estar re-escribiendo el código de manera repetitiva. A continuación se mostrará el código, donde se cambiará las ultimas lineas para aplicar uno u otro método y la 'variable `initial_state` según lo anteriormente explicado, por otro lado se descimentará y comentará lo interesado según el método a usar. en el main.
En el código se mostrar será el utilizado para **Búsqueda en anchura usando árbol** y ***Profundidad 1 (1,2,3,4,5,0,7,8,6)***.

Para el estudio comparativo entre distintos métodos usaré un estado de profundidad 15

- Profundidad 15: (1,0,8,5,2,3,4,7,6)




```python
import os
import time
import psutil

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
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


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

# ______________________________________________________________________________


class Invented(Problem):

    def __init__(self, initial_state= (1,2,3,4,5,0,7,8,6), goal_state= (1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """
        Inicializa una instancia del problema del 8 puzzle.
        :param initial_state: El estado inicial del problema.
        :param goal_state: El estado objetivo del problema.
        """
        super().__init__(initial_state, goal_state)

    def actions(self, state):
        """
        Devuelve las acciones legales desde el estado `state`.
        :param state: El estado actual.
        :return: Lista de acciones legales desde el estado actual.
        """
        actions = []
        empty_pos = state.index(0)  # posición del hueco

        if empty_pos % 3 > 0:
            actions.append('left')
        if empty_pos % 3 < 2:
            actions.append('right')
        if empty_pos // 3 > 0:
            actions.append('up')
        if empty_pos // 3 < 2:
            actions.append('down')

        return actions

    def result(self, state, action):
        """
        Devuelve el estado resultante de aplicar la acción `action` en el estado `state`.
        :param state: El estado actual.
        :param action: La acción a aplicar.
        :return: El estado resultante de aplicar la acción.
        """
        new_state = list(state)  # Hacemos una copia del estado actual
        empty_pos = state.index(0)  # posición del hueco

        if action == 'left':
            new_pos = empty_pos - 1
        elif action == 'right':
            new_pos = empty_pos + 1
        elif action == 'up':
            new_pos = empty_pos - 3
        elif action == 'down':
            new_pos = empty_pos + 3

        # Intercambiamos el valor en la posición `empty_pos` con el valor en la posición `new_pos`
        new_state[empty_pos], new_state[new_pos] = new_state[new_pos], new_state[empty_pos]

        return tuple(new_state)

    def goal_test(self, state):
        """
        Determina si el estado `state` es un estado objetivo.
        :param state: El estado a comprobar.
        :return: True si el estado es un estado objetivo, False en caso contrario.
        """
        return state == self.goal


if __name__ == '__main__':
    process = psutil.Process(os.getpid())
    print('\nMemory usage initially: %s (%.2f%%)\n' % (humanbytes(process.memory_info().rss), process.memory_percent() ) )

    problem: Problem = Invented()

    start = time.process_time()
    # Refers to the ime the CPU was busy processing the program’s instructions.
    # The time spent waiting for other task to complete (like I/O operations) is not included in the CPU time.
    solution: Optional[Node] = breadth_first_tree_search(problem)
    #solution: Optional[Node] = breadth_first_graph_search(problem)
    #solution: Optional[Node] = depth_first_graph_search(problem)
    #solution: Optional[Node] = depth_first_tree_search(problem)
    elapsed = time.process_time() - start

    if solution is not None:
        print("Nodos expandidos: ", Statistics().get_amount())
        print("Profundidad de la solución: ", solution.depth)
        print("Nodos:", solution.path(), sep='\n\t')
        print("Acciones:", solution.solution(), sep='\n\t')

    print('\nMemory usage finally: %s (%.2f%%)\n' % (humanbytes(process.memory_info().rss), process.memory_percent() ) )
    print('CPU Execution time: %.6f seconds' % elapsed)

```
~~~

#######################################################
Profundidad 1
Búsqueda en anchura usando árbol
#######################################################

Memory usage initially: 61.01 MB (0.38%)

Nodos expandidos:  3

Profundidad de la solución:  1

Nodos:
	[<Node (1, 2, 3, 4, 5, 0, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down']

Memory usage finally: 61.01 MB (0.38%)

CPU Execution time: 0.000072 seconds

#######################################################
Profundidad 2
Búsqueda en anchura usando árbol
#######################################################

Memory usage initially: 61.02 MB (0.38%)

Nodos expandidos:  15

Profundidad de la solución:  2

Nodos:
	[<Node (1, 2, 3, 4, 0, 6, 7, 5, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 0, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down', 'right']

Memory usage finally: 61.02 MB (0.38%)

CPU Execution time: 0.000139 seconds

#######################################################
Profundidad 3
Búsqueda en anchura usando árbol
#######################################################

Memory usage initially: 60.99 MB (0.38%)

Nodos expandidos:  31

Profundidad de la solución:  3

Nodos:
	[<Node (1, 2, 3, 0, 5, 6, 4, 7, 8)>, <Node (1, 2, 3, 4, 5, 6, 0, 7, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 0, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down', 'right', 'right']

Memory usage finally: 60.99 MB (0.38%)

CPU Execution time: 0.000175 seconds

#######################################################
Profundidad 4
Búsqueda en anchura usando árbol
#######################################################

Memory usage initially: 61.02 MB (0.38%)

Nodos expandidos:  42

Profundidad de la solución:  4

Nodos:
	[<Node (1, 3, 0, 4, 2, 5, 7, 8, 6)>, <Node (1, 0, 3, 4, 2, 5, 7, 8, 6)>, <Node (1, 2, 3, 4, 0, 5, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 0, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['left', 'down', 'right', 'down']

Memory usage finally: 61.02 MB (0.38%)

CPU Execution time: 0.000370 seconds

#######################################################
Profundidad 5
Búsqueda en anchura usando árbol
#######################################################

Memory usage initially: 61.00 MB (0.38%)

Nodos expandidos:  134

Profundidad de la solución:  5

Nodos:
	[<Node (2, 0, 3, 1, 4, 6, 7, 5, 8)>, <Node (0, 2, 3, 1, 4, 6, 7, 5, 8)>, <Node (1, 2, 3, 0, 4, 6, 7, 5, 8)>, <Node (1, 2, 3, 4, 0, 6, 7, 5, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 0, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['left', 'down', 'right', 'down', 'right']

Memory usage finally: 61.00 MB (0.38%)

CPU Execution time: 0.000717 seconds

#######################################################
Profundidad 15
Búsqueda en anchura usando árbol
#######################################################

Memory usage initially: 61.13 MB (0.38%)

Nodos expandidos:  6874757

Profundidad de la solución:  15

Nodos:
	[<Node (1, 0, 8, 5, 2, 3, 4, 7, 6)>, <Node (1, 2, 8, 5, 0, 3, 4, 7, 6)>, <Node (1, 2, 8, 0, 5, 3, 4, 7, 6)>, <Node (0, 2, 8, 1, 5, 3, 4, 7, 6)>, <Node (2, 0, 8, 1, 5, 3, 4, 7, 6)>, <Node (2, 8, 0, 1, 5, 3, 4, 7, 6)>, <Node (2, 8, 3, 1, 5, 0, 4, 7, 6)>, <Node (2, 8, 3, 1, 0, 5, 4, 7, 6)>, <Node (2, 0, 3, 1, 8, 5, 4, 7, 6)>, <Node (0, 2, 3, 1, 8, 5, 4, 7, 6)>, <Node (1, 2, 3, 0, 8, 5, 4, 7, 6)>, <Node (1, 2, 3, 4, 8, 5, 0, 7, 6)>, <Node (1, 2, 3, 4, 8, 5, 7, 0, 6)>, <Node (1, 2, 3, 4, 0, 5, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 0, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down', 'left', 'up', 'right', 'right', 'down', 'left', 'up', 'left', 'down', 'down', 'right', 'up', 'right', 'down']

Memory usage finally: 105.14 MB (0.66%)

CPU Execution time: 37.727990 seconds

#######################################################
Profundidad 1
Búsqueda en anchura usando grafo
#######################################################

Memory usage initially: 61.13 MB (0.38%)

Nodos expandidos:  1

Profundidad de la solución:  1

Nodos:
	[<Node (1, 2, 3, 4, 5, 0, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down']

Memory usage finally: 60.96 MB (0.38%)

CPU Execution time: 0.000128 seconds

#######################################################
Profundidad 2
Búsqueda en anchura usando grafo
#######################################################

Memory usage initially: 61.09 MB (0.38%)

Nodos expandidos:  5

Profundidad de la solución:  2

Nodos:
	[<Node (1, 2, 3, 4, 0, 6, 7, 5, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 0, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down', 'right']

Memory usage finally: 61.09 MB (0.38%)

CPU Execution time: 0.000096 seconds

#######################################################
Profundidad 3
Búsqueda en anchura usando grafo
#######################################################

Memory usage initially: 61.04 MB (0.38%)

Nodos expandidos:  9

Profundidad de la solución:  3

Nodos:
	[<Node (1, 2, 3, 0, 5, 6, 4, 7, 8)>, <Node (1, 2, 3, 4, 5, 6, 0, 7, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 0, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down', 'right', 'right']

Memory usage finally: 61.04 MB (0.38%)

CPU Execution time: 0.000119 seconds

#######################################################
Profundidad 4
Búsqueda en anchura usando grafo
#######################################################

Memory usage initially: 60.99 MB (0.38%)

Nodos expandidos:  10

Profundidad de la solución:  4

Nodos:
	[<Node (1, 3, 0, 4, 2, 5, 7, 8, 6)>, <Node (1, 0, 3, 4, 2, 5, 7, 8, 6)>, <Node (1, 2, 3, 4, 0, 5, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 0, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['left', 'down', 'right', 'down']

Memory usage finally: 60.99 MB (0.38%)

CPU Execution time: 0.000119 seconds

#######################################################
Profundidad 5
Búsqueda en anchura usando grafo
#######################################################

Memory usage initially: 61.11 MB (0.38%)

Nodos expandidos:  22

Profundidad de la solución:  5

Nodos:
	[<Node (2, 0, 3, 1, 4, 6, 7, 5, 8)>, <Node (0, 2, 3, 1, 4, 6, 7, 5, 8)>, <Node (1, 2, 3, 0, 4, 6, 7, 5, 8)>, <Node (1, 2, 3, 4, 0, 6, 7, 5, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 0, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['left', 'down', 'right', 'down', 'right']

Memory usage finally: 61.11 MB (0.38%)

CPU Execution time: 0.000323 seconds

#######################################################
Profundidad 15
Búsqueda en anchura usando grafo
#######################################################

Memory usage initially: 61.04 MB (0.38%)

Nodos expandidos:  4361

Profundidad de la solución:  15

Nodos:
	[<Node (1, 0, 8, 5, 2, 3, 4, 7, 6)>, <Node (1, 2, 8, 5, 0, 3, 4, 7, 6)>, <Node (1, 2, 8, 0, 5, 3, 4, 7, 6)>, <Node (0, 2, 8, 1, 5, 3, 4, 7, 6)>, <Node (2, 0, 8, 1, 5, 3, 4, 7, 6)>, <Node (2, 8, 0, 1, 5, 3, 4, 7, 6)>, <Node (2, 8, 3, 1, 5, 0, 4, 7, 6)>, <Node (2, 8, 3, 1, 0, 5, 4, 7, 6)>, <Node (2, 0, 3, 1, 8, 5, 4, 7, 6)>, <Node (0, 2, 3, 1, 8, 5, 4, 7, 6)>, <Node (1, 2, 3, 0, 8, 5, 4, 7, 6)>, <Node (1, 2, 3, 4, 8, 5, 0, 7, 6)>, <Node (1, 2, 3, 4, 8, 5, 7, 0, 6)>, <Node (1, 2, 3, 4, 0, 5, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 0, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down', 'left', 'up', 'right', 'right', 'down', 'left', 'up', 'left', 'down', 'down', 'right', 'up', 'right', 'down']

Memory usage finally: 62.68 MB (0.39%)

CPU Execution time: 1.019572 seconds

#######################################################
Profundidad 1
Búsqueda en profundidad usando grafo
#######################################################

Memory usage initially: 61.02 MB (0.38%)

Nodos expandidos:  1

Profundidad de la solución:  1

Nodos:
	[<Node (1, 2, 3, 4, 5, 0, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down']

Memory usage finally: 61.03 MB (0.38%)

CPU Execution time: 0.000068 seconds

#######################################################
Profundidad 2
Búsqueda en profundidad usando grafo
#######################################################

Memory usage initially: 60.99 MB (0.38%)

Nodos expandidos:  2

Profundidad de la solución:  2

Nodos:
	[<Node (1, 2, 3, 4, 0, 6, 7, 5, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 0, 8)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down', 'right']

Memory usage finally: 60.99 MB (0.38%)

CPU Execution time: 0.000110 seconds

#######################################################
Profundidad 3
Búsqueda en profundidad usando grafo
#######################################################

Memory usage initially: 61.22 MB (0.38%)

Proceso interrumpido tras mas de 2 minutos de proceso... se considera que no encuentra la solución.

#######################################################
Profundidad 4
Búsqueda en profundidad usando grafo
#######################################################

Memory usage initially: 61.09 MB (0.38%)

Nodos expandidos:  30

Profundidad de la solución:  30

Nodos:
	[<Node (1, 3, 0, 4, 2, 5, 7, 8, 6)>, <Node (1, 3, 5, 4, 2, 0, 7, 8, 6)>, <Node (1, 3, 5, 4, 2, 6, 7, 8, 0)>, <Node (1, 3, 5, 4, 2, 6, 7, 0, 8)>, <Node (1, 3, 5, 4, 0, 6, 7, 2, 8)>, <Node (1, 0, 5, 4, 3, 6, 7, 2, 8)>, <Node (1, 5, 0, 4, 3, 6, 7, 2, 8)>, <Node (1, 5, 6, 4, 3, 0, 7, 2, 8)>, <Node (1, 5, 6, 4, 3, 8, 7, 2, 0)>, <Node (1, 5, 6, 4, 3, 8, 7, 0, 2)>, <Node (1, 5, 6, 4, 0, 8, 7, 3, 2)>, <Node (1, 0, 6, 4, 5, 8, 7, 3, 2)>, <Node (1, 6, 0, 4, 5, 8, 7, 3, 2)>, <Node (1, 6, 8, 4, 5, 0, 7, 3, 2)>, <Node (1, 6, 8, 4, 5, 2, 7, 3, 0)>, <Node (1, 6, 8, 4, 5, 2, 7, 0, 3)>, <Node (1, 6, 8, 4, 0, 2, 7, 5, 3)>, <Node (1, 0, 8, 4, 6, 2, 7, 5, 3)>, <Node (1, 8, 0, 4, 6, 2, 7, 5, 3)>, <Node (1, 8, 2, 4, 6, 0, 7, 5, 3)>, <Node (1, 8, 2, 4, 6, 3, 7, 5, 0)>, <Node (1, 8, 2, 4, 6, 3, 7, 0, 5)>, <Node (1, 8, 2, 4, 0, 3, 7, 6, 5)>, <Node (1, 0, 2, 4, 8, 3, 7, 6, 5)>, <Node (1, 2, 0, 4, 8, 3, 7, 6, 5)>, <Node (1, 2, 3, 4, 8, 0, 7, 6, 5)>, <Node (1, 2, 3, 4, 8, 5, 7, 6, 0)>, <Node (1, 2, 3, 4, 8, 5, 7, 0, 6)>, <Node (1, 2, 3, 4, 0, 5, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 0, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down', 'down', 'left', 'up', 'up', 'right', 'down', 'down', 'left', 'up', 'up', 'right', 'down', 'down', 'left', 'up', 'up', 'right', 'down', 'down', 'left', 'up', 'up', 'right', 'down', 'down', 'left', 'up', 'right', 'down']

Memory usage finally: 61.09 MB (0.38%)

CPU Execution time: 0.000282 seconds

#######################################################
Profundidad 5
Búsqueda en profundidad usando grafo
#######################################################

Memory usage initially: 61.11 MB (0.38%)

Nodos expandidos:  330

Profundidad de la solución:  323

Nodos: Se descartan para la presentación.

Acciones: Se descartan para la presentación.

Memory usage finally: 61.11 MB (0.38%)

CPU Execution time: 0.009761 seconds


#######################################################
Profundidad 15
Búsqueda en profundidad usando grafo
#######################################################

Memory usage initially: 61.06 MB (0.38%)

Nodos expandidos:  46201

Profundidad de la solución:  42387

Nodos: Se descartan para la presentación.

Acciones: Se descartan para la presentación.

Memory usage finally: 86.67 MB (0.54%)

CPU Execution time: 130.345431 seconds


#######################################################
Profundidad 1
Búsqueda en profundidad usando árbol
#######################################################

Memory usage initially: 61.21 MB (0.38%)

Nodos expandidos:  1

Profundidad de la solución:  1

Nodos:
	[<Node (1, 2, 3, 4, 5, 0, 7, 8, 6)>, <Node (1, 2, 3, 4, 5, 6, 7, 8, 0)>]

Acciones:
	['down']

Memory usage finally: 61.21 MB (0.38%)

CPU Execution time: 0.000060 seconds

#######################################################
Profundidad 2
Búsqueda en profundidad usando árbol
#######################################################

Proceso "killed" por excesivo consumo de RAM.

#######################################################
Profundidad 3
Búsqueda en profundidad usando árbol
#######################################################

Proceso "killed" por excesivo consumo de RAM.

#######################################################
Profundidad 4
Búsqueda en profundidad usando árbol
#######################################################

Proceso "killed" por excesivo consumo de RAM.

#######################################################
Profundidad 5
Búsqueda en profundidad usando árbol
#######################################################

Proceso "killed" por excesivo consumo de RAM.

#######################################################
Profundidad 15
Búsqueda en profundidad usando árbol
#######################################################

Proceso "killed" por excesivo consumo de RAM.

~~~

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

---

---

---





## Exposición de los datos obtenidos.

A continuación, se procede a realizar una tabla comparativa de cada uno de los procesos de búsqueda según su profundidad.

### **Búsqueda en anchura usando árbol**

| | **Profundidad 1** | **Profundidad 2** | **Profundidad 3** | **Profundidad 4** | **Profundidad 5** | **Profundidad 15** |
| --- | --- | --- | --- | --- | ---| ---|
| Nodos expandidos | 3 | 15 | 32 | 43 | 134 | 6874757 |
| Profundidad de la solución | 1 | 2 | 3 | 4 | 5 | 15 |
| Memoria usada finalmente | 61.01 MB | 61.02 MB | 60.99 MB | 61.02 MB | 61.00 MB | 105.14 MB |
| Tiempo de ejecución de CPU | 0.000072 seconds |  0.000139 seconds | 0.000175 seconds | 0.000370 seconds | 0.000717 seconds | 37.727990 seconds  |


### **Búsqueda en anchura usando grafo**

| | **Profundidad 1** | **Profundidad 2** | **Profundidad 3** | **Profundidad 4** | **Profundidad 5** | **Profundidad 15** |
| --- | --- | --- | --- | --- | ---| ---|
| Nodos expandidos | 1 | 5 | 9 | 10 | 22 | 4361 |
| Profundidad de la solución | 1 | 2 | 3 | 4 | 5 | 15 |
| Memoria usada finalmente | 60.96 MB | 61.09 MB | 61.04 MB | 60.99 MB | 61.11 MB | 62.68 MB |
| Tiempo de ejecución de CPU | 0.000128 seconds | 0.000096 seconds | 0.000119 seconds | 0.000119 seconds | 0.000323 seconds | 1.019572 seconds |
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


### **Búsqueda en profundidad usando grafo**

| | **Profundidad 1** | **Profundidad 2** | **Profundidad 4** | **Profundidad 5** | **Profundidad 15** |
| --- | --- | --- | --- | ---| ---|
| Nodos expandidos | 1 | 2 | 30 | 330 | 46201 |
| Profundidad de la solución | 1 | 2 | 30 | 323 | 42387 |
| Memoria usada finalmente | 61.03 MB | 60.99 MB | 61.09 MB | 61.11 MB | 86.67 MB |
| Tiempo de ejecución de CPU | 0.000068 second | 0.000110 seconds | 0.000282 seconds | 0.009761 seconds | 130.345431 seconds |

El proceso en profundidad 3 se ha descartado tras estar mas de 2 minutos ejecutándose, dando por "killed" el proceso.


### **Búsqueda en profundidad usando árbol**

| | **Profundidad 1** |
| --- | --- |
| Nodos expandidos | 1 |
| Profundidad de la solución | 1 |
| Memoria usada finalmente | 61.21 MB |
| Tiempo de ejecución de CPU | 0.000060 seconds |

En los procesos en profundidad 2,3,4,5 y 15 se ha procedido a hacer "kill" en su ejecución debido al excesivo consumo de RAM que estaban produciendo.

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

---

---

---

---

---

---

---


## Comparación entre modelos de búsqueda.

Se procede a realizar una tabla comparativa para ver cual es el mejor modelo de trabajo para este caso usando la profundidad 15.
Se descarta directamente `Búsqueda en profundidad usando árbol` debido a que este proceso ha entrado en **bucle infinito** y por tanto su funcionalidad para este caso es nula.

| | **Búsqueda en anchura usando árbol** | <span style="color:red">**Búsqueda en anchura usando grafo**</span> | **Búsqueda en profundidad usando grafo** |
| --- | --- | --- | --- |
| Nodos expandidos | 6874757 | **4361** | 46201 |
| Profundidad de la solución | **15** | **15** | 42387 |
| Memoria usada finalmente | 105.14 MB | **62.68 MB** | 86.67 MB |
| Tiempo de ejecución de CPU | 37.727990 seconds | **1.019572 seconds** | 130.345431 seconds |

Cabe destacar de los datos obtenidos la siguiente reflexión:

El tipo de búsqueda mas efectivo, aun desconociendo el coste tanto en Nodos expandidos, profundidad de la solución, memoria utilizada finalmente, como tiempo de ejecución en su búsqueda, es ´***Búsqueda en anchura usando grafo***´. Se puede destacar que `Búsqueda en anchura usando árbol` encuentra de la misma manera la solución mas efectiva, pero realizando un mayor consumo en tiempo, nodos expandidos y memoria.

---

## Conclusión

Como conclusión al estudio de datos obtenidos y las reflexiones realizadas, se puede determinar que el método de **Búsqueda en anchura usando grafo** es el más eficiente de los 4 que se han realizado.
