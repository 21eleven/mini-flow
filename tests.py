"""
This script builds and runs test graphs with miniflow.
"""

from miniflow import *

# Test 1 -  feed forward thru add and multiply nodes

x, y, z = Input(), Input(), Input()

f = Mul(x, y, z)
k = Add(x, f)

feed_dict = {x:4, y:5, z:10}

sorted_nodes = topological_sort(feed_dict)
graph = sorted_nodes
forward_and_backward_propagation(graph)
output = k

print("\nTest 1 - feed forward thru add and multiply nodes") 
print("( {} * {} * {} ) + {} = {}".format(x.value, y.value, z.value, x.value, output.value))

# Test 2 - feed forward thru linear unit

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)
feed_dict = {
    inputs:  [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
forward_and_backward_propagation(graph)
output = f
print("\nTest 2 - feed forward thru linear unit")
print("Inputs: {}\nWeights: {}\nBias: {}".format(feed_dict[inputs], feed_dict[weights], feed_dict[bias]))
print("Output: {}".format(output.value))
inputs, weights, bias = Input(), Input(), Input()

# Test 3 - feed forward through linear and sigmoid layers

h = Linear(inputs, weights, bias)
f = Sigmoid(h)

X_ = np.array([[-1., -2.],[-1.,-2.]])
W_ = np.array([[2., -3.],[2., -3.]])
b_ = np.array([-3., -5])

feed_dict = {inputs: X_, weights: W_, bias: b_}

graph = topological_sort(feed_dict)
forward_and_backward_propagation(graph)
output = f
print("\nTest 3 - feed forward through linear and sigmoid layers")
print("Inputs: \n{}\nWeights: \n{}\nBias: \n{}".format(feed_dict[inputs], feed_dict[weights], feed_dict[bias]))
print("Output: \n{}".format(output.value))

# Test 4 - MSE cost function

y, a = Input(), Input()
cost=MSE(y, a)

y_ = np.array([1,2,3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y:y_,a:a_}
graph = topological_sort(feed_dict)
forward_and_backward_propagation(graph)
print("\nTest 4 - MSE cost function")
print("y: \n{}\ny-hat:\n{}\n".format(y_,a_))
print("Expecting 23.4166666667:")
print(cost.value)

# Test 5 - backpropagate gradients through linear and sigmoid layers

X, W, b = Input(), Input(), Input()
y = Input()
f = Linear(X, W, b)
a = Sigmoid(f)
cost = MSE(y, a)

X_ = np.array([[-1., -2.],[-1.,-2.]])
W_ = np.array([[2.],[3.]])
b_ = np.array([-3.])
y_ = np.array([1., 2.])

feed_dict = {
    X: X_,
    y: y_,
    W: W_,
    b: b_,
}

graph = topological_sort(feed_dict)
forward_and_backward_propagation(graph)
gradients = [t.gradients[t] for t in [X, y, W, b]]
print("\nTest 5 - backpropagate gradients through linear and sigmoid layers")
print("Inputs: \n{}\nWeights: \n{}\nBias: \n{}".format(feed_dict[X], feed_dict[W], feed_dict[b]))
print("""Gradients backpropagated, expected:
[array([[ -3.34017280e-05,  -5.01025919e-05],
       [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833],
       [ 1.9999833]]), array([[  5.01028709e-05],
       [  1.00205742e-04]]), array([ -5.01028709e-05])]""")
print("Gradients backpropagated, actual:\n{}".format(gradients))


