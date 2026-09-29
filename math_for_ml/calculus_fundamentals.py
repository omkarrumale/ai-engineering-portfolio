"""
Calculus for ML — Numerical Implementations
Topics: Derivatives, Gradient Descent, Partial Derivatives, Chain Rule
Purpose: Understanding mathematical foundations through code
"""

# Taking the derivatives of 2,3,4
def f_task1(x):
    return x**2

h = 0.0001

for x in [2, 3, 4]:
    numerical = (f_task1(x+h) - f_task1(x)) / h
    analytical = 2*x
    print(f"x={x} | Numerical: {numerical:.4f} | Analytical: {analytical}")


# Finding Gradient
w = 10
learning_rate = 0.1

for i in range(50):
    gradient = 2*(w-3)
    w = w-learning_rate*gradient

    if i%10==0:
        print(f"Iteration {i} | w = {w:.4f}")
print(f"final w = {w:.4f}")


# partial derivation
def f_partial(x, y):
    return x**2 + 2*x*y + y**2
h = 0.0001
partial_x = (f_partial(2+h, 3) - f_partial(2, 3)) / h

partial_y = (f_partial(2, 3+h) - f_partial(2, 3)) / h

print(f"∂f/∂x at (2,3) = {partial_x}")
print(f"∂f/∂y at (2,3) = {partial_y}")

# finding analytical answer
x=2
y=3
analy_x = 2*x+2*y
analy_y = 2*x+2*y

print(f"The analytical answer of x = {analy_x}\nThe analytical answer of y = {analy_y}")

# Chain rule
# f(x) = (3x + 2)⁴

# 1.compute derivation at x = 1
def f_chain(x):
    return (3*x + 2)**4

x = 1
h = 0.0001

# Numerical derivative
numerical = (f_chain(x + h) - f_chain(x)) / h

# Analytical derivative
analytical = 12 * (3*x + 2)**3

print(f"Numerical derivative at x=1:{numerical:.2f}")
print(f"Analytical derivative at x=1:{analytical}")
