
class Node(object):

    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        self.value = None

    def forward(self):
        raise NotImplemented

class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = sum([n.value for n in self.inbound_nodes])

class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 1
        for n in self.inbound_nodes:
            self.value *= n.value

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        self.value=sum([(self.inbound_nodes[0].value[i]*self.inbound_nodes[1].value[i]) for i in range(0,len(self.inbound_nodes[0].value))])+self.inbound_nodes[2].value

def forward_pass(output_node, sorted_nodes):
    for n in sorted_nodes:
        n.forward()
    return output_node.value


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
