# Resolución de problemas basados en búsquedas

## Tarea1: Experimentación de búsqueda en anchura y profundidad

El objetivo de está tarea es el de entender el código del programa adjunto y poder experimentar la búsqueda en anchura y profundidad.

Es necesario que se ejecute el código y se tomen los datos de tiempo de ejecución, memoria consumida y cantidad de nodos expandidos para:

- Búsqueda en anchura usando árbol.
- Búsqueda en anchura usando grafo.
- Búsqueda en profundidad usando grafo.
- Búsqueda en profundidad usando árbol.

## Solución

La tarea a ejecutar es dado un numero inicial A menor que 100, mediante las operaciones (-10,-1,+1,+10) debe de llegar a un numero B objetivo.

En este caso el A= 89, B=43

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
    """
    Example of implementation of the problem.
    The states are two digits (from 10 to 99)
    The actions is to add or subtract some digit (+1, -1, +10, -10)
    """

    def __init__(self, initial=89, goal=43):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['+1', '+10', '-1', '-10']

        if state == 99:
            possible_actions.remove('+1')
        if state >= 90:
            possible_actions.remove('+10')

        if state < 20:
            possible_actions.remove('-10')
        if state == 10:
            possible_actions.remove('-1')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        tmp = {
            '+1': 1,
            '+10': 10,
            '-1': -1,
            '-10': -10
        }

        return state + tmp[action]

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal


if __name__ == '__main__':
    process = psutil.Process(os.getpid())
    print('\nMemory usage initially: %s (%.2f%%)\n' % (humanbytes(process.memory_info().rss), process.memory_percent() ) )

    problem: Problem = Invented()

    start = time.process_time()
    # Refers to the ime the CPU was busy processing the program’s instructions.
    # The time spent waiting for other task to complete (like I/O operations) is not included in the CPU time.
    for i in range (3):
        print("##########################################################")
        print("# Ejercicio", i+1)
        print("##########################################################")
        if i==0:
            print("Búsqueda en anchura usando árbol.")
            solution: Optional[Node] = breadth_first_tree_search(problem)
        elif i==1:
            print("Búsqueda en anchura usando grafo.")
            solution: Optional[Node] = breadth_first_graph_search(problem)
        elif i==2:
            print("Búsqueda en profundidad usando grafo.")
            solution: Optional[Node] = depth_first_graph_search(problem)
        #elif i==3:
            #print("Búsqueda en profundidad usando grafo.")
            #print("Se procede a cancelar su ejecución por entrar en bucle infinito")
            #solution: Optional[Node] = depth_first_tree_search(problem)
        elapsed = time.process_time() - start
    
        if solution is not None:
            print("Nodos expandidos: ", Statistics().get_amount())
            print("Profundidad de la solucion: ", solution.depth)
            print("Nodos:", solution.path(), sep='\n\t')
            print("Acciones:", solution.solution(), sep='\n\t')
    
            print('\nMemory usage finally: %s (%.2f%%)\n' % (humanbytes(process.memory_info().rss), process.memory_percent() ) )
        
        
        
    print("##########################################################")
    print("# Ejercicio 4")
    print("##########################################################")
        
    print("Búsqueda en profundidad usando grafo.")
    print("Se procede a cancelar su ejecución por entrar en bucle infinito.")
```

    
    Memory usage initially: 61.09 MB (0.38%)
    
    ##########################################################
    # Ejercicio 1
    ##########################################################
    Búsqueda en anchura usando árbol.
    Nodos expandidos:  47569
    Profundidad de la solucion:  9
    Nodos:
    	[<Node 89>, <Node 90>, <Node 91>, <Node 92>, <Node 93>, <Node 83>, <Node 73>, <Node 63>, <Node 53>, <Node 43>]
    Acciones:
    	['+1', '+1', '+1', '+1', '-10', '-10', '-10', '-10', '-10']
    
    Memory usage finally: 63.41 MB (0.40%)
    
    ##########################################################
    # Ejercicio 2
    ##########################################################
    Búsqueda en anchura usando grafo.
    Nodos expandidos:  47631
    Profundidad de la solucion:  9
    Nodos:
    	[<Node 89>, <Node 90>, <Node 91>, <Node 92>, <Node 93>, <Node 83>, <Node 73>, <Node 63>, <Node 53>, <Node 43>]
    Acciones:
    	['+1', '+1', '+1', '+1', '-10', '-10', '-10', '-10', '-10']
    
    Memory usage finally: 63.16 MB (0.40%)
    
    ##########################################################
    # Ejercicio 3
    ##########################################################
    Búsqueda en profundidad usando grafo.
    Nodos expandidos:  47662
    Profundidad de la solucion:  30
    Nodos:
    	[<Node 89>, <Node 79>, <Node 69>, <Node 59>, <Node 49>, <Node 39>, <Node 29>, <Node 19>, <Node 18>, <Node 17>, <Node 16>, <Node 15>, <Node 14>, <Node 13>, <Node 12>, <Node 11>, <Node 21>, <Node 31>, <Node 41>, <Node 51>, <Node 61>, <Node 71>, <Node 81>, <Node 91>, <Node 92>, <Node 93>, <Node 83>, <Node 73>, <Node 63>, <Node 53>, <Node 43>]
    Acciones:
    	['-10', '-10', '-10', '-10', '-10', '-10', '-10', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '+10', '+10', '+10', '+10', '+10', '+10', '+10', '+10', '+1', '+1', '-10', '-10', '-10', '-10', '-10']
    
    Memory usage finally: 63.16 MB (0.40%)
    
    ##########################################################
    # Ejercicio 4
    ##########################################################
    Búsqueda en profundidad usando grafo.
    Se procede a cancelar su ejecución por entrar en bucle infinito.
    

---

## Exposición de los datos obtenidos.

Se procede a realizar una tabla comparativa para ver cual es el mejor modelo de trabajo para este caso.
Se descarta directamente `Búsqueda en profundidad usando grafo` debido a que este proceso ha entrado en **bucle infinito** y por tanto su funcionalidad para este caso es nula.

| | **Búsqueda en anchura usando árbol** | **Búsqueda en anchura usando grafo** | **Búsqueda en profundidad usando grafo** |
| --- | --- | --- | --- |
| Nodos expandidos | ***47569*** | 47631 |47662 |
| Profundidad de la solución | ***9*** | ***9*** | 30 |
| Memoria usada finalmente | 63.41 MB | ***63.16 MB*** | ***63.16 MB*** |

Cabe destacar de los datos obtenidos las siguientes reflexiones:
- No aparece en los datos sacados por la consola de Python el tiempo de ejecución de cada uno de los tipos de búsqueda, aunque si es importante indicar que el tiempo de todos ellos fueron inmediatos. Por lo que se puede tomar como un dato despreciable.
- De la misma manera se puede indicar en lo relativo a los nodos expandidos ya que el ahorro de nodos expandidos entre la opción mas optima y la menos optima en este campo de estudio, es solo de un **0.1955%**. Por lo que esta opción como decisiva para su elección no se considera la mas decisiva.
- Relativo al apartado de la profundidad de la solución, pasa lo contrario que en el apartado anterior, ya que el método de `Búsqueda en profundidad usando grafo` utiliza 3 veces mas la profundidad de la solución que los otros dos métodos, este campo de estudio se puede considerar determinante ya que la diferencia del menos con el mayor es de mas de **300%**, quedando la  opción de `Búsqueda en profundidad usando grafo` como **DESCARTADA**.
- Relativo a la memoria utilizada finalmente ocurre algo parecido como en los nodos expandidos, la diferencia es que en vez de ser del 0.1955%, en este caso es del **0.3958%** siendo de esta manera un poco mas determinante a la opción entre elegir `Búsqueda en anchura usando grafo` o `Búsqueda en anchura usando árbol`. 


---

## Conclusión

Como conclusión al estudio de datos obtenidos y las reflexiones realizadas, se puede determinar que el método de **Búsqueda en anchura usando grafo** es el más eficiente de los 4 que se han realizado.
