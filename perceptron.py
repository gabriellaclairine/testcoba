# You can run online via https://repl.it/languages/python3
from random import choice
from numpy import array, dot, random

unit_step = lambda x: 0 if x < 0 else 1

# Fourth data is bias constan
training_data = [
    (array([1,2,1,1]), 1),
    (array([3,1,1,1]), 0) ]

#w = random.rand(4)
w = array([1. , -1.,  2.,  -2.])
errors = []
eta = 0.2   # Learning rate
n = 100    # number of iteration

for i in range(n):
	# Online Learning 
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x

# Testing
for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))
