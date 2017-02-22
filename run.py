"""
This script builds and runs a graph with miniflow.
"""

from miniflow import *

# Example 1 -  Simple feed forward thru add and multiply nodes

x, y, z = Input(), Input(), Input()

f = Mul(x, y, z)
k = Add(x, f)

feed_dict = {x:4, y:5, z:10}

sorted_nodes = topological_sort(feed_dict)
graph = sorted_nodes
output = forward_pass(k, graph)

print("Example 1 -  Simple feed forward thru add and multiply nodes") 
print("Miniflow: ( {} * {} * {} ) + {} = {}".format(x.value, y.value, z.value, x.value, output))


inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    inputs:  [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
print("Example 2 - feed forward thru linear unit")
print("Inputs: {}\nWeights: {}\nBias: {}".format(feed_dict[inputs], feed_dict[weights], feed_dict[bias]))
print("Output: {}".format(output))
