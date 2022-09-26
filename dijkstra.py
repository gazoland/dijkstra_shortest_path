import sys
import numpy as np


class PriorityQueue:
    """
    Represents a circular priority queue object.

    Attributes:
        capacity: int, the maximum number of elements the queue can have
        head: int, the index of the first element in the queue
        tail: int, the index of the last element in the queue
        total_elements: int, the total number of elements in the queue in a given moment
        elements: numpy.ndarray, sequence of elements in the queue
    """

    def __init__(self, capacity: int):
        """
        Constructor of the PQ object.

        :param capacity: int, the maximum number of elements the queue can have
        """
        self.capacity = capacity
        self.head = 0
        self.tail = -1
        self.total_elements = 0
        self.elements = np.empty(capacity, dtype=object)

    def __is_empty(self):
        """Checks whether the PQ has any element. :returns: bool, whether the PQ has any element."""
        return self.total_elements == 0

    def __is_full(self):
        """Checks whether the PQ is full. :returns: bool, whether the PQ is full."""
        return self.total_elements == self.capacity

    def __move_elements(self, index_range_lower_limit, index_range_upper_limit):
        """
        Moves every element of the PQ one position behind, between two range limits.

        :param index_range_lower_limit: int, lower limit of the moving elements. It is where a new element will be added
        :param index_range_upper_limit: int, upper limit of the moving elements. It is the index of the last element
        """
        # Aux variable to help calculate the range if it goes through the circular characteristic of the PQ
        range_adjust = self.capacity if index_range_lower_limit > index_range_upper_limit else 0
        # Loop through elements in reverse, starting at one position after the last element of the PQ
        for j in range(index_range_upper_limit + 1, index_range_lower_limit - range_adjust, -1):
            if j < 0:
                j += self.capacity
            elif j >= self.capacity - 1:
                j -= self.capacity
            # Make next element equals to the previous one
            self.elements[j] = self.elements[j - 1]

    def __increase_head(self):
        """Increases the head attribute of the priority queue."""
        if self.head < self.capacity - 1:
            self.head += 1
        else:
            self.head = 0

    def __increase_tail(self):
        """Increases the tail attribute of the priority queue."""
        if self.tail < self.capacity - 1:
            self.tail += 1
        else:
            self.tail = 0

    def add(self, node, node_exists: bool = False, existing_node_index=None):
        """
        Inserts a DijkstraNode object into the PQ.

        :param node: DijkstraNode, represents the node used in the Dijkstra algorithm
        :param node_exists: bool, whether the PQ already contains a node that represents the same node being added
        :param existing_node_index: int, the index on the PQ of the node that represents the same node being added
        :returns: bool, whether the operation was successful
        """
        if not self.__is_full():
            end_of_elements = self.tail
            # If the node being added is already in the PQ, the new node will have a shorter path, and we do not need
            # to analyze the elements after the existing node.
            if node_exists:
                # end_of_elements is the last element before the existing node
                end_of_elements = existing_node_index - 1
            # Now, we need to find the correct index of the node to be to inserted
            idx_corrector = 0
            # Assume the node will be inserted at the end
            idx_to_insert = 0 if end_of_elements == self.capacity - 1 else end_of_elements + 1
            # Loop through all existing elements:
            for i in range(self.total_elements):
                if i + self.head >= self.capacity:  # checks if index will get out of range
                    idx_corrector = self.capacity
                # If that element has a longer path than the new node being added:
                if node.path_distance < self.elements[i + self.head - idx_corrector].path_distance:
                    # We found the index we must insert the new node
                    idx_to_insert = i + self.head - idx_corrector
                    # Now we must move all subsequent elements one spot behind to make room for the new node.
                    # If idx_to_insert is not the same index of the element representing the existing node,
                    # (end_of_elements + 1) -> keeping in mind the circular possibilty:
                    if idx_to_insert != end_of_elements + 1 and \
                            not (idx_to_insert == 0 and end_of_elements == self.capacity - 1):
                        # Move those elements
                        self.__move_elements(idx_to_insert, end_of_elements)
                    break
            # Add node at found index
            self.elements[idx_to_insert] = node
            # If the DijkstraNode was not previously in the PQ:
            if not node_exists:
                self.__increase_tail()
                self.total_elements += 1
            return True

        raise Exception("Queue overflow.")

    def get_first(self):
        """Gets the head element and removes it from the queue. :returns: the element at the head of the queue."""
        if not self.__is_empty():
            first = self.elements[self.head]
            self.__increase_head()
            self.total_elements -= 1
            return first
        print("Queue is empty.")
        return None


class Stack:
    """
    Representes a stack data structure, a sequence of objects with LIFO characteristics.

    Attributes:
        capacity: int, the total number of elements the stack can have
        top: int, indicates the index of the LIFO element of the sequence
        elements: numpy.ndarray, sequence (array) of elements in the stack
    """
    def __init__(self, capacity: int):
        """
        Constructor of the Stack object.
        :param capacity: int, the maximum number of elements the stack can have
        """
        self.capacity = capacity
        self.top = -1
        self.elements = np.empty(capacity, dtype=object)

    def __is_empty(self):
        """Checks whether the stack has any element. :returns: bool, whether the stack has any element."""
        return self.top == -1

    def __is_full(self):
        """Checks whether the stack is full. :returns: bool, whether the stack is full."""
        return self.top == self.capacity - 1

    def add(self, node):
        """
        Adds an element to the stack.

        :param node: DijkstraNode, representing a node in the Dijkstra algorithm application
        :returns: bool, whether the operation was successful
        """
        if not self.__is_full():
            self.top += 1
            self.elements[self.top] = node
            return True
        print("Stack is full.")
        return False

    def remove(self):
        """Removes the top element from the stack. :returns: bool, whether the operation was successful."""
        if not self.__is_empty():
            self.top -= 1
            return True
        print("Stack is empty")
        return False

    def get_top(self):
        """Gets the top element and removes it from the stack. :returns: top element from the stack."""
        if not self.__is_empty():
            first = self.elements[self.top]
            self.remove()
            return first
        print("Stack is empty.")
        return None


class Vertex:
    """
    Represents a vertex in a graph.

    Attributes:
        label: str, the "name" of the vertex. It will be treated as an identifier
        neighbours: list, contains Edge objects with information about each edge connected to the vertex
        visited: bool, whether the vertex was already analyzed by the Dijkstra algorithm
    """
    def __init__(self, label: str):
        """
        Constructor of the Vertex object.

        :param label: str, the "name" or identifier of this vertex
        """
        self.label = label
        self.neighbours = []
        self.visited = False

    def add_neighbour(self, edge):
        """
        Adds an Edge object that is connected to that vertex.

        :param edge: Edge, contains information of the adjacent vertex and the weight of that edge
        :returns: bool, acknolewdgment
        """
        self.neighbours.append(edge)
        return True

    def show_neighbours(self):
        """
        Prints to the user which vertices that vertex is connected to

        :return: None, message is printed to the user
        """
        for n in self.neighbours:
            print("{} - Distance: {}".format(n.vertex.label, n.distance))


class Edge:
    """
    Represents the edges of a graph. For every vertex, there is an edge that reaches another vertex.

    Attributes:
        vertex: Vertex, the adjacent vertex that the edge connects to
        distance: int, the weight attributed to that edge on the graph, connecting both vertices
    """
    def __init__(self, vertex, distance: int):
        """
        Constructor of the Edge object
        :param vertex: Vertex, the adjacent vertex that the edge connects to
        :param distance: int, the weight attributed to that edge on the graph, connecting both vertices
        """
        self.vertex = vertex
        self.distance = distance


class DijkstraNode:
    """
    Represents a node used in the Dijkstra algorithm, with information of the corresponding vertex on the graph.

    Attributes:
        vertex: Vertex object, representing a given vertex from a graph
        path_distance: int, the total distance from the starting point to that vertex
        previous: DijkstraNode, the vertex that immediately precedes this node itself
    """
    def __init__(self, vertex, previous):
        """
        Constructor of the DijkstraNode object.

        :param vertex: Vertex, representing a vertex on a graph
        :param previous: DijkstraNode, representing the DijkstraNode node that precedes the current node
        """
        self.vertex = vertex
        self.path_distance = sys.maxsize
        self.previous = previous


class DijkstraAlgo:
    """
    Represents the application of a Dijkstra Shortest Path algorithm.

    Attributes:
        start: Vertex object, representing the starting position on a graph
        graph: Graph object, representing an actual graph of the problem
        queue: PriorityQueue object, the queue used to analyze nodes on the Dijkstra algorithm
        stack: Stack object, stores the vertices/nodes already visited by the algorithm
    """
    def __init__(self, starting_vertex, graph, nodes):
        """
        Constructor of the Dijkstra algorithm object.

        :param starting_vertex: Vertex, the starting point of the graph
        :param graph: Graph, representing an actual graph of the problem
        :param nodes: list, with all the vertices on the graph
        """
        self.start = starting_vertex
        self.graph = graph
        self.queue = PriorityQueue(len(nodes))
        self.stack = Stack(len(nodes))

    def __is_dijkstra_node_in_queue(self, dijkstra_node):
        """
        Checks if new DijkstraNode is in the PriorityQueue

        :param dijkstra_node: DijkstraNode, representing the vertex we are looking for in the PQ
        :returns:
            - already_in: bool, whether the dijkstra_node is in the PQ
            - queue_index: int, the index of that dijkstra_node in the PQ
        """
        idx_corrector = 0
        already_in = False
        queue_index = self.queue.head  # Not used outside of the method if already_in returns False
        for node_index in range(self.queue.total_elements):
            if node_index + self.queue.head >= self.queue.capacity:  # Checks if index will get out of range
                idx_corrector = self.queue.capacity
            queue_index = node_index + self.queue.head - idx_corrector  # Index of PQ
            # If that element at queue_index is the current dijkstra_node we are searching for:
            if self.queue.elements[queue_index].vertex.label == dijkstra_node.vertex.label:
                already_in = True
                break
        return already_in, queue_index

    def __return_route(self, destination_d_node):
        """
        Prints to the user the optimal path to the desired destination.

        :param destination_d_node: DijkstraNode, representing the desired destination vertex
        :returns: None
        """
        # Create a list that will receive the DijkstraNode objects involved in the optimal path, from end to start
        vertex_sequence = list()
        # Get the previous node from which we got to the current node
        current_previous = destination_d_node.previous
        # While we don't reach the starting node:
        while self.stack.top != -1:
            # Get the top of the stack out
            top = self.stack.get_top()
            # If that node is the node that originates the current node, add it to the list
            if current_previous == top:
                vertex_sequence.append(top)
                current_previous = top.previous
        print("To get to {}, go on the following route:".format(destination_d_node.vertex.label))
        # Printing the nodes in the list in reverse order, from start to end
        for i in range(len(vertex_sequence) - 1, -1, -1):
            print(vertex_sequence[i].vertex.label)
        print(destination_d_node.vertex.label)
        print("The total distance is: {}".format(destination_d_node.path_distance))
        return

    def find_route(self, destination_vertex):
        """
        Looks for the optimal route to a given destination.

        :param destination_vertex: Vertex, the desired destination of the graph
        :returns: None, the route will be printed to the user by the __return_route method
        """
        # Create initial node to be added into the PQ and set its path_distance to 0
        initial_node = DijkstraNode(self.start, previous=None)
        initial_node.path_distance = 0
        self.queue.add(initial_node)
        # While there is an element in the PQ we will continue to execute these steps
        while self.queue.total_elements != 0:
            # Get the node at the head of the PQ
            queue_head_node = self.queue.elements[self.queue.head]
            # Mark the node as visited
            queue_head_node.vertex.visited = True
            # If the head of the PQ is our destination node, that means that no other subsequent path can possibly have
            # a lower total path than the one we currently have, because the PQ is ordered based on total distance from
            # the start. So, we can stop iterating and return the answer.
            if destination_vertex.label == queue_head_node.vertex.label:
                self.__return_route(queue_head_node)
                return
            # For every neighbour of that node
            for n in queue_head_node.vertex.neighbours:  # n is an Edge object
                # If node was already visited, we don't need to analyze it again. The current path will be longer
                if not n.vertex.visited:
                    # We either add it to the queue if it's not in it yet, or we update the path_distance if it's lower
                    # Create a DijkstraNode of that vertex and calculate the path distance
                    d_node = DijkstraNode(vertex=n.vertex, previous=queue_head_node)
                    d_node.path_distance = n.distance + d_node.previous.path_distance
                    # Check if it is already in the PQ
                    already_in, queue_node_index = self.__is_dijkstra_node_in_queue(d_node)
                    # If d_node is already in the PQ AND d_node.path < queue[q_n_index].path:
                    if already_in and d_node.path_distance < self.queue.elements[queue_node_index].path_distance:
                        # Add it again to the queue. The existing node with longer path will be replaced.
                        # And we only need to analyze the PQ from head to the element before the existing d_node.
                        # If the new path is still the longest within this PQ range, it will be added to the same index.
                        # If not, it will be added accordingly.
                        self.queue.add(d_node, node_exists=True, existing_node_index=queue_node_index)

                    # If not in the PQ, add it normally
                    if not already_in:
                        self.queue.add(d_node)
            # Get the head node out of the PQ and add it to the Stack object
            head_of_queue = self.queue.get_first()
            self.stack.add(head_of_queue)


class RomanianCitiesGraph:
    """Represents the weighted graph of the Romanian cities from the problem."""

    # Creating all vertices
    arad = Vertex("arad")
    zerind = Vertex("zerind")
    oradea = Vertex("oradea")
    sibiu = Vertex("sibiu")
    timisoara = Vertex("timisoara")
    lugoj = Vertex("lugoj")
    mehadia = Vertex("mehadia")
    dobreta = Vertex("dobreta")
    craiova = Vertex("craiova")
    rimnicu = Vertex("rimnicu vilcea")
    fagaras = Vertex("fagaras")
    pitesti = Vertex("pitesti")
    bucharest = Vertex("bucharest")
    giurgiu = Vertex("giurgiu")
    urziceni = Vertex("urziceni")
    hirsova = Vertex("hirsova")
    eforie = Vertex("eforie")
    vaslui = Vertex("vaslui")
    iasi = Vertex("iasi")
    neamt = Vertex("neamt")

    # Adding all edges to the vertices
    arad.add_neighbour(Edge(zerind, 75))
    arad.add_neighbour(Edge(sibiu, 140))
    arad.add_neighbour(Edge(timisoara, 118))
    zerind.add_neighbour(Edge(arad, 75))
    zerind.add_neighbour(Edge(oradea, 71))
    oradea.add_neighbour(Edge(sibiu, 151))
    oradea.add_neighbour(Edge(zerind, 71))
    sibiu.add_neighbour(Edge(oradea, 151))
    sibiu.add_neighbour(Edge(arad, 140))
    sibiu.add_neighbour(Edge(rimnicu, 80))
    sibiu.add_neighbour(Edge(fagaras, 99))
    timisoara.add_neighbour(Edge(arad, 118))
    timisoara.add_neighbour(Edge(lugoj, 111))
    lugoj.add_neighbour(Edge(timisoara, 111))
    lugoj.add_neighbour(Edge(mehadia, 70))
    mehadia.add_neighbour(Edge(lugoj, 70))
    mehadia.add_neighbour(Edge(dobreta, 75))
    dobreta.add_neighbour(Edge(mehadia, 75))
    dobreta.add_neighbour(Edge(craiova, 120))
    craiova.add_neighbour(Edge(dobreta, 120))
    craiova.add_neighbour(Edge(rimnicu, 146))
    craiova.add_neighbour(Edge(pitesti, 138))
    rimnicu.add_neighbour(Edge(craiova, 146))
    rimnicu.add_neighbour(Edge(pitesti, 97))
    rimnicu.add_neighbour(Edge(sibiu, 80))
    fagaras.add_neighbour(Edge(sibiu, 99))
    fagaras.add_neighbour(Edge(bucharest, 211))
    pitesti.add_neighbour(Edge(craiova, 138))
    pitesti.add_neighbour(Edge(rimnicu, 97))
    pitesti.add_neighbour(Edge(bucharest, 101))
    bucharest.add_neighbour(Edge(pitesti, 101))
    bucharest.add_neighbour(Edge(fagaras, 211))
    bucharest.add_neighbour(Edge(giurgiu, 90))
    bucharest.add_neighbour(Edge(urziceni, 85))
    giurgiu.add_neighbour(Edge(bucharest, 90))
    urziceni.add_neighbour(Edge(bucharest, 85))
    urziceni.add_neighbour(Edge(hirsova, 98))
    urziceni.add_neighbour(Edge(vaslui, 142))
    hirsova.add_neighbour(Edge(urziceni, 98))
    hirsova.add_neighbour(Edge(eforie, 86))
    eforie.add_neighbour(Edge(hirsova, 86))
    vaslui.add_neighbour(Edge(urziceni, 142))
    vaslui.add_neighbour(Edge(iasi, 92))
    iasi.add_neighbour(Edge(vaslui, 92))
    iasi.add_neighbour(Edge(neamt, 87))
    neamt.add_neighbour(Edge(iasi, 87))


# All cities from the problem
cities = ["arad", "zerind", "oradea", "sibiu", "timisoara", "lugoj", "mehadia", "dobreta", "craiova", "rimnicu vilcea",
          "fagaras", "pitesti", "bucharest", "giurgiu", "urziceni", "hirsova", "eforie", "vaslui", "iasi", "neamt"]


def test():
    graph = RomanianCitiesGraph()
    dij = DijkstraAlgo(graph.sibiu, graph, cities)
    dij.find_route(graph.mehadia)


if __name__ == "__main__":
    test()
