# Define your function first, then use it
def f(x):
    return x**2

# Then compute derivative
h = 0.0001
x = 2
derivative = (f(x+h) - f(x)) / h
print(f"Numerical derivative at x=2: {derivative:.4f}")
print(f"Analytical answer (2x): {2*x}")