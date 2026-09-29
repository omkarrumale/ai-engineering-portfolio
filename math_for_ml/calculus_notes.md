# Calculus for ML — Notes

## 1. power rule 
- f(x) = x^n
- f'(x) = n*x^n-1
- for constants 
- f'(5) = 0

## 2. Chain Rule
- When a function inside another function that is a composite function you use chain rule
- fromula/notation
dy/dx = df/du * du/dx
- Chain rule is nothing but derivation of outer X * derivation of inner x

## 3. Quotient rule
- A quotient rule tells you how to differentiate a function that is a ratio of two function
formula:-
         u'v - uv' / v^2

## 4. product rule
- keep the first differentiate second 
- keep the second differentiate first
formula :-
          u'v  + uv'

## 5. Derivatives
- It tells instantaneous rate of change at a specific point
-notation
df/df or f'(x)

## 6. eˣ derivative
- eˣ is special because its derivative is itself.
- f(x) = eˣ → f'(x) = eˣ
- No other function has this property.
- Bigger IT gate --> faster it grows

- Why it is used in ML
1. Smooth derivation
2. Stable gradient


## 7. ln(x) derivative
- Log base E inverse of E to the power X
- In these growth slows down as X increases
- Opposite behavior of exponential
- f(x) = ln(x) → f'(x) = 1/x

## 8. Sigmoid derivation
- Sigmoid used in neural network to squish any values between zero and one
- Sigmoid = Probability converter
- Output close to 1 -> Yes/True
- Output close 0 -> No/False
- σ(x) = 1/(1+e⁻ˣ)
σ'(x) = σ(x)(1-σ(x))
Efficient because σ(x) already computed in forward pass — reused in backward pass

## 9. second derivative
- Second derivative tells you the curvature whether the function meaning upward or downward
- f''(x) > 0 The curve bent upward U shape (minimum)
- f''(x) < 0 The curve bent downward N shape (maximum)
- f''(x) = 0 Inconclusive -> Could be anything

## 10. Convexity
- Convex function → f''(x) > 0 everywhere → bowl shape → one minimum
- Non-convex → f''(x) < 0 somewhere → multiple hills and valleys
- Convex → gradient descent guaranteed to find global minimum
- Non-convex → gradient descent might get stuck in local minimum
- Neural network loss functions are non-convex

## 11. Limits
- Limit ask what value is the function approaching
- Limit can exist even a function is undefined
-notation
lim  f(x)
x->a

## 12. Taylor Series
- any smooth function can be approximated a nearer point using its derivative at that point
- First order taylor (Linear approximation)
f(x) = f(a) + f'(a)(x-a)
- Second order: f(x) ≈ f(a) + f'(a)(x-a) + f''(a)/2·(x-a)²
ML use: mathematical foundation of gradient descent and learning rate

## 13. Partial derivatives
= It mean that change one variable keep the other constant and differentiate
notation
| Notation                        | Meaning            |
| ------------------------------- | ------------------ |
| (\frac{\partial f}{\partial x}) | derivative w.r.t x |
| (\frac{\partial f}{\partial y}) | derivative w.r.t y |
- We can say that when differentiating with respect to one variable treat all other variable as constant

## 14. Mixed Partial Derivatives
- Differentiate with respect to one variable then another
- ∂²f/∂x∂y = differentiate w.r.t y first, then x
- Clairaut's theorem: ∂²f/∂x∂y = ∂²f/∂y∂x for smooth functions
- Order of differentiation doesn't matter
- This is why Hessian matrix is always symmetric

## 15. Gradient vector
- Gradient vector collects all the partial derivatives
- It tells which direction to move
- How fast function is increasing
- ∇f = [∂f/∂x, ∂f/∂y, ...]
Gradient points steepest INCREASE
Gradient descent moves -∇f (opposite) for steepest DECREASE

## 16. Directional derivatives
- It tells you the rate of change of function in any chosen direction
- formula:-
           Du​f=∇f⋅u

## 17. jacobian matrix
- A Jacobian is an m*n matrix of all partial derivatives
- The Jacobian is a matrix that tells you how all output changes with respect to all input 

## 18. Hessian matrix
- The Hessian collects all the second order partial derivatives into one matrix
- It tells you how the slope is changing not just slope also the curvature

## 19. Computational graphs
- A computational graph breaks any function into simple operation as nodes
- Each node does one simple thing add multiply square etc....

# 20. Backpropagation
- Backpropagation also known as backward pass compute derivative by moving right to left using chain rule
- Each node asks one question how much I contribute to the final error and pass that information backward to the previous node
- To calculate the back propagation you have to calculate the forward pass first 

## 21. Local vs global minimum, saddle points
- Local minimum-> Lowest point in the small region only
- Global minimum-> Lowest point in the entire function
- Saddle point-> Not minimum,not maximum flat in One Direction

## 22. Gradient descent — batch, SGD, mini-batch
- Gradient descent->Gradient descent is a method where we keep adjusting values step by step in the opposite direction of the slope to reduce the error.
- Batch gradient-> Use all data for compute gradient
- SGD gradient-> Use one data point at a time
- Mini batch gradient-> Use a small batches

## 23.Learning rate behavior
- Learning rate or behavior is basically how the learning process changes depending on the learning rate value
- Learning rate controls
  - How fast you move
  - How stable training is
- There are three main behaviors

1. Too small learning rate-> Very tiny step slow process
2. Good learning rate-> Balance steps good movement
3. Two large learning rate-> Big jumps overshooting

- Batch GD → accurate but too slow for large data
- SGD → fast but noisy and unstable
- Mini-batch → balance of both → industry standard






