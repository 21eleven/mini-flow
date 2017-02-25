import numpy as np

class Node(object):

    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        self.outbound_nodes = []
        self.gradients = {}
        self.value = None

    def forward(self):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self : 0}
        for n in self.outbound_nodes:
            gradient_from_edge = n.gradients[self]
            self.gradients[self] += gradient_from_edge * 1

class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = sum([n.value for n in self.inbound_nodes])

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            gradient_from_edge = n.gradients[self]
            for i in range(0,len(self.inbound_nodes)):
                self.gradients[self.inbound_nodes[i]] += gradient_from_edge * 1

class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 1
        for n in self.inbound_nodes:
            self.value *= n.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            gradient_from_edge = n.gradients[self]
            for i in range(0,len(self.inbound_nodes)):
                self.gradients[self.inbound_nodes[i]] += gradient_from_edge * 1

class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        self.value = np.dot(self.inbound_nodes[0].value, self.inbound_nodes[1].value) + self.inbound_nodes[2].value

    def backward(self):
        # a partial derivative for each of the input nodes
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # pull gradient from each edge connected to an output node
        for n in self.outbound_nodes:

            # partial wrt to this node
            gradient_from_edge = n.gradients[self]

            # partial wrt to this node's inputs
            self.gradients[self.inbound_nodes[0]] +=  np.dot(gradient_from_edge,self.inbound_nodes[1].value.T)

            # partial wrt to this node's weights
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T,gradient_from_edge)

            # partial wrt to this node's bias
            self.gradients[self.inbound_nodes[2]] += np.sum(gradient_from_edge, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, input_node):
        Node.__init__(self, [input_node])

    def _sigmoid(self, x):
        return 1./(1.+np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        sigmoid_derivative = self.value * (1 - self.value)
        for n in self.outbound_nodes:
            gradient_from_edge = n.gradients[self]
            for i in range(0,len(self.inbound_nodes)):
                self.gradients[self.inbound_nodes[i]] += sigmoid_derivative*gradient_from_edge


class MSE(Node):
    """
    Calculates the mean squared error
    """
    def __init__(self, y, a):
        # Where y is expected output and a is actual output
        Node.__init__(self, [y, a])

    def forward(self):
        # Reshape to avoid potential matrix/vector broadcast errors
        y = self.inbound_nodes[0].value.reshape(-1,1)
        a = self.inbound_nodes[1].value.reshape(-1,1)

        self.m = len(y)
        self.diff = y-a
        sum_sqr_err = np.sum(np.square(self.diff))
        self.value = sum_sqr_err/self.m

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff

def forward_and_backward_propagation(graph):
    for n in graph:
        n.forward()
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]


def topological_sort(feed_dict):
    """
    Sort nodes in topological order using Kahn's Algorithm.
    """
    input_nodes = [n for n in feed_dict.keys()]
    G = {}
    nodes = [n for n in input_nodes]

    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in':set(), 'out':set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in':set(), 'out':set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    linear_order = []
    S = set(input_nodes)

    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        linear_order.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges to add to S
            if len(G[m]['in']) == 0:
                S.add(m)

    return linear_order
