# Resolución de problemas basados en búsquedas

## Tarea 2: Números

En esta tarea es necesario que utilices el código de la tarea 1. Crea un problema cuyos estados sean números de tres cifras y las posibles acciones sean sumar o restar una unidad a alguna de las cifras (+1, +10, +100, -1, -10, -100).

El problema en concreto tendrá una serie de números tabú que deberían de ser imposibles de alcanzar.

Es decir, la definición del problema es la siguiente:

Estado inicial: 789
Estado Objetivo: 269
Números tabú: 244, 253, 254, 343, 344, 353, 778, 779, 679, 689
Acciones: +1, +10, +100, -1, -10, -100


## Solución

La tarea a ejecutar es dado un numero inicial A menor que 1000 y mayor que 99, mediante las operaciones (-100, -10, -1, +1, +10, +100) debe de llegar a un numero B objetivo menor que 1000 y mayor que 99 objetivo.

En este caso el A= 789, B=269
No se puede llegar a los siguientes numeros: 244, 253, 254, 343, 344, 353, 778, 779, 679, 689

Se va a proceder a modificar el código para que los 4 posibles búsquedas las haga a la vez y se muestre en el código una vez ejecutado. Para ello se pondrá un for in range 4. Se añadirán cabeceras para separar cada uno de los resultados


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
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
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
    """
    Example of implementation of the problem.
    The states are two digits (from 10 to 99)
    The actions is to add or subtract some digit (+1, -1, +10, -10)
    """

    def __init__(self, initial=789, goal=269):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        actions = [1, 10, 100, -1, -10, -100]
        tabu = [244, 253, 254, 343, 344, 353, 778, 779, 679, 689]
        possible_actions = []

        for action in actions:
            tmp = state + action
            if 100 <= tmp <= 999 and tmp not in tabu:
                possible_actions.append(action)

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        return state + action

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal


if __name__ == '__main__':
    for i in range(3):
        process = psutil.Process(os.getpid())
        print('\nMemory usage initially: %s (%.2f%%)\n' % (humanbytes(process.memory_info().rss), process.memory_percent() ) )
    
        problem: Problem = Invented()
    
        start = time.process_time()
        # Refers to the ime the CPU was busy processing the program’s instructions.
        # The time spent waiting for other task to complete (like I/O operations) is not included in the CPU time.
        if i==0:
            print("############################################")
            print("Parte ", i+1)
            print("############################################")
            solution: Optional[Node] = breadth_first_tree_search(problem)
        if i==1:
            print("############################################")
            print("Parte ", i+1)
            print("############################################")
            solution: Optional[Node] = breadth_first_graph_search(problem)
        if i==2:
            print("############################################")
            print("Parte ", i+1)
            print("############################################")
            solution: Optional[Node] = depth_first_graph_search(problem)
        #if i==3:
            #print("############################################")
            #print("Parte ", i+1)
            #print("############################################")
            #solution: Optional[Node] = depth_first_tree_search(problem)
        elapsed = time.process_time() - start
    
        if solution is not None:
            print("Nodos expandidos: ", Statistics().get_amount())
            print("Profundidad de la solucion: ", solution.depth)
            print("Nodos:", solution.path(), sep='\n\t')
            print("Acciones:", solution.solution(), sep='\n\t')
    
        print('\nMemory usage finally: %s (%.2f%%)\n' % (humanbytes(process.memory_info().rss), process.memory_percent() ) )
        print('CPU Execution time: %.6f seconds' % elapsed)
    print("Fin de la ejecución.")


        
    print("##########################################################")
    print("# Parte 4")
    print("##########################################################")
        
    print("Búsqueda en profundidad usando grafo.")
    print("Se procede a cancelar su ejecución por entrar en bucle infinito.")
```

    
    Memory usage initially: 61.05 MB (0.38%)
    
    ############################################
    Parte  1
    ############################################
    Nodos expandidos:  1684758
    Profundidad de la solucion:  9
    Nodos:
    	[<Node 789>, <Node 790>, <Node 780>, <Node 770>, <Node 769>, <Node 669>, <Node 569>, <Node 469>, <Node 369>, <Node 269>]
    Acciones:
    	[1, -10, -10, -1, -100, -100, -100, -100, -100]
    
    Memory usage finally: 100.46 MB (0.63%)
    
    CPU Execution time: 15.275921 seconds
    
    Memory usage initially: 100.46 MB (0.63%)
    
    ############################################
    Parte  2
    ############################################
    Nodos expandidos:  1685223
    Profundidad de la solucion:  9
    Nodos:
    	[<Node 789>, <Node 790>, <Node 780>, <Node 770>, <Node 769>, <Node 669>, <Node 569>, <Node 469>, <Node 369>, <Node 269>]
    Acciones:
    	[1, -10, -10, -1, -100, -100, -100, -100, -100]
    
    Memory usage finally: 99.54 MB (0.63%)
    
    CPU Execution time: 0.014522 seconds
    
    Memory usage initially: 99.54 MB (0.63%)
    
    ############################################
    Parte  3
    ############################################
    Nodos expandidos:  1685359
    Profundidad de la solucion:  128
    Nodos:
    	[<Node 789>, <Node 788>, <Node 688>, <Node 588>, <Node 488>, <Node 388>, <Node 288>, <Node 188>, <Node 178>, <Node 168>, <Node 158>, <Node 148>, <Node 138>, <Node 128>, <Node 118>, <Node 108>, <Node 107>, <Node 106>, <Node 105>, <Node 104>, <Node 103>, <Node 102>, <Node 101>, <Node 100>, <Node 200>, <Node 190>, <Node 180>, <Node 170>, <Node 160>, <Node 150>, <Node 140>, <Node 130>, <Node 120>, <Node 220>, <Node 219>, <Node 209>, <Node 309>, <Node 299>, <Node 399>, <Node 499>, <Node 599>, <Node 699>, <Node 709>, <Node 708>, <Node 608>, <Node 508>, <Node 408>, <Node 407>, <Node 307>, <Node 297>, <Node 197>, <Node 196>, <Node 186>, <Node 176>, <Node 166>, <Node 156>, <Node 146>, <Node 136>, <Node 126>, <Node 125>, <Node 124>, <Node 123>, <Node 122>, <Node 222>, <Node 212>, <Node 211>, <Node 311>, <Node 301>, <Node 291>, <Node 281>, <Node 271>, <Node 261>, <Node 251>, <Node 241>, <Node 231>, <Node 331>, <Node 330>, <Node 329>, <Node 328>, <Node 327>, <Node 227>, <Node 217>, <Node 216>, <Node 215>, <Node 214>, <Node 314>, <Node 304>, <Node 294>, <Node 194>, <Node 184>, <Node 174>, <Node 164>, <Node 154>, <Node 144>, <Node 143>, <Node 243>, <Node 233>, <Node 333>, <Node 323>, <Node 423>, <Node 413>, <Node 403>, <Node 393>, <Node 383>, <Node 283>, <Node 273>, <Node 263>, <Node 363>, <Node 362>, <Node 352>, <Node 342>, <Node 442>, <Node 432>, <Node 532>, <Node 522>, <Node 512>, <Node 502>, <Node 492>, <Node 482>, <Node 472>, <Node 471>, <Node 461>, <Node 451>, <Node 450>, <Node 350>, <Node 349>, <Node 249>, <Node 259>, <Node 269>]
    Acciones:
    	[-1, -100, -100, -100, -100, -100, -100, -10, -10, -10, -10, -10, -10, -10, -10, -1, -1, -1, -1, -1, -1, -1, -1, 100, -10, -10, -10, -10, -10, -10, -10, -10, 100, -1, -10, 100, -10, 100, 100, 100, 100, 10, -1, -100, -100, -100, -1, -100, -10, -100, -1, -10, -10, -10, -10, -10, -10, -10, -1, -1, -1, -1, 100, -10, -1, 100, -10, -10, -10, -10, -10, -10, -10, -10, 100, -1, -1, -1, -1, -100, -10, -1, -1, -1, 100, -10, -10, -100, -10, -10, -10, -10, -10, -1, 100, -10, 100, -10, 100, -10, -10, -10, -10, -100, -10, -10, 100, -1, -10, -10, 100, -10, 100, -10, -10, -10, -10, -10, -10, -1, -10, -10, -1, -100, -1, -100, 10, 10]
    
    Memory usage finally: 99.54 MB (0.63%)
    
    CPU Execution time: 0.007187 seconds
    Fin de la ejecución.
    ##########################################################
    # Parte 4
    ##########################################################
    Búsqueda en profundidad usando grafo.
    Se procede a cancelar su ejecución por entrar en bucle infinito.


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


---


## Exposición de los datos obtenidos.

Se procede a realizar una tabla comparativa para ver cual es el mejor modelo de trabajo para este caso.
Se descarta directamente `Búsqueda en profundidad usando árbol` debido a que este proceso ha entrado en **bucle infinito** y por tanto su funcionalidad para este caso es nula.

| | **Búsqueda en anchura usando árbol** | <span style="color:red">**Búsqueda en anchura usando grafo**</span> | **Búsqueda en profundidad usando grafo** |
| --- | --- | --- | --- |
| Nodos expandidos | ***1684758*** | 1685223 | 1685359 |
| Profundidad de la solución | ***9*** | ***9*** | 128 |
| Memoria usada finalmente | 100.46 MB | ***99.54 MB*** | ***99.54 MB*** |
| Tiempo de ejecución de CPU | 15.275921 seconds | *0.014522 seconds* | ***0.007187 seconds*** | 

Cabe destacar de los datos obtenidos las siguientes reflexiones:

- Se puede indicar en lo relativo a los nodos expandidos ya que el ahorro de nodos expandidos entre la opción mas optima y la menos optima en este campo de estudio, es solo de un **0.0357%**. Por lo que esta opción como decisiva para su elección no se considera la mas decisiva.
- Relativo al apartado de la profundidad de la solución, pasa lo contrario que en el apartado anterior, ya que el método de `Búsqueda en profundidad usando grafo` utiliza 3 veces mas la profundidad de la solución que los otros dos métodos, este campo de estudio se puede considerar determinante ya que la diferencia del menos con el mayor es de mas de **1433%**, quedando la  opción de `Búsqueda en profundidad usando grafo` como **DESCARTADA**.
- Relativo a la memoria utilizada finalmente ocurre algo parecido como en los nodos expandidos, la diferencia es que en vez de ser del 0.1955%, en este caso es del **1.2085%** siendo de esta manera un poco mas determinante a la opción entre elegir `Búsqueda en anchura usando grafo` o `Búsqueda en anchura usando árbol`. 
- A modo de re-afirmación, se puede concluir que el tiempo de ejecución diferenciando entre `Búsqueda en anchura usando árbol` y `Búsqueda en anchura usando grafo`, cabe destacar que en `Búsqueda en anchura usando árbol` usa muchísimo mas tiempo de ejecución que `Búsqueda en anchura usando grafo`.



---

## Conclusión

Como conclusión al estudio de datos obtenidos y las reflexiones realizadas, se puede determinar que el método de **Búsqueda en anchura usando grafo** es el más eficiente de los 4 que se han realizado.
