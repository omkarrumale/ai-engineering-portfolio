import numpy as np

# Basic Probability
# bag has 5 red, 3 blue, 2 green balls
# calculate P(red), P(blue), P(green)
# calculate P(red or blue)
# calculate P(not green)
# Total balls 
total = 10
red = 5
blue = 3
green = 2

p_red = red / total
p_blue = blue / total
p_green = green / total

# Addition rule — mutually exclusive
p_red_or_blue = p_red + p_blue

# Complement rule
p_not_green = 1 - p_green


print(f"Probability of (red) = {p_red}")
print(f"Probability of (blue) = {p_blue}")
print(f"Probability of (green) = {p_green}")
print(f"P(red or blue) = {p_red_or_blue}")
print(f"P(not green) = {p_not_green}")

# ---------Bayes theorem --------------------------------------

# Machine A → 60% production, 4% defective
# Machine B → 40% production, 6% defective
# item is defective → P(from machine B)?
# calculate P(defective) first
# then apply Bayes formula

p_A = 0.6
p_B = 0.4
p_DA = 0.04
p_DB = 0.06 
'''So here we just need to calculate the p_BD
so formula look like p_BD = p_DB *p_B / p_D so we dont have p_D so we will calculate it first and then use it in formula  '''

p_D = p_DA * p_A + p_DB * p_B
print(F"So the value of p_D = {p_D}")

'''applying p_D in the main bayes formula formula'''
p_BD = p_DB * p_B / p_D
print(f"Therefoe the value of p_DB = {p_BD} which means ~50%  chances that is came from machine B")


# -------------Probability Distributions---------------------------
# Binomial(n=10, p=0.3) → P(X=4), mean, variance
# Normal(mu=70, sigma=15) → generate 1000 samples
#                         → verify mean and std
#                         → check 68% rule
# Poisson(lambda=5) → P(X=3)

# 1.binomial

'''Here we are going to calculate the number of success in n independent trials'''
'''Binomial formula = p(x = k) = (c(n,k) * p^k * (1-p)^(n-k))'''
n = 10
k = 4
p = 0.3

'''So here firsst we are going to calculate (n,k)
formula = c(n,k) = n!/k! * (n-k)!'''

c = (10*9*8*7) / (4*3*2*1)
print(f"The value of c = {c}") 

p_x = (c * (p**k) * (1-p)**(n-k))
print(f"Therefore the value of p_x = {p_x:.3f}")

'''Let's calculate mean and variance '''
mean = n*p
print(f"Therefore the mean  = {mean}")

variance = n*p*(1-p)
print(f"Therefore the variance = {variance:.2f}")

# 2.Normal
'''So How we are gonna normal distribution it means that continuous probability distribution that is symmetrical and bell shape'''

mu = 70
sigma = 15

# Generate 1000 samples from Normal(70, 15)
samples = np.random.normal(mu, sigma, 1000)

# Verify parameters
print(f"Sample mean = {np.mean(samples):.2f}  (expected {mu})")
print(f"Sample std  = {np.std(samples):.2f}  (expected {sigma})")

# Verify 68% rule
within_1std = np.sum((samples > mu-sigma) & (samples < mu+sigma))
percentage = within_1std / 1000 * 100
print(f"% within 1 std = {percentage:.1f}%  (expected ~68%)")

# 3.Poisson
''' So now we are going to calculate poisson distribution It means a number of events occur in a fixed interval of time/space'''

Lambda = 5
x = 3
e = np.e

'''Formula = p(x) = lambda^x * e^-lambda / x!'''
p = ((Lambda**x) * (e**(-Lambda)) / (3*2*1))
print(f"Therefore the value of p = {p:.3f}\nIt means that there are around 14% chances")


# ----------------Statistics--------------------------

data = np.array([23, 45, 67, 12, 89, 34, 56, 78, 90, 23, 45, 200])
# mean, median, mode
# variance, std
# IQR and outlier detection

# 1.mean,median,mode
mean = np.mean(data)
print(f"The mean of the data = {mean}")

median = np.median(data)
print(f"The median of the data = {median}")

values, counts = np.unique(data, return_counts=True)
mode = values[np.argmax(counts)]
print(f"Mode = {mode}")

# 2.variance,std
variance = np.var(data)
print(f"The variance of data = {variance:2f}")

std = np.std(data)
print(f"The std of data = {std:.2f}")

# 3.IQR
Q1 = np.quantile(data,0.25)
Q3 = np.quantile(data,0.75)

IQR = Q3-Q1

lower = Q1-1.5*IQR
upper = Q3+1.5*IQR

outliers = data[(data<lower) | (data>upper)]
print("Outlier range:", lower , 'to' , upper)
print(f"outliers = {outliers}")


# ------------Hypothesis Testing------------------------
# sample = np.random.normal(33, 7, 49)
# H0: mu = 30, H1: mu > 30
# calculate Z manually
# use scipy to verify
# print decision

''' So here we are Or do a hypothesis testing 
H0-> Null hypothesis
H1-> Alternative hypothesis'''
''' Here for H0 = MU = 30
    AND H1 = mu > 30'''

'''formula = (x_bar - mu) / (sigma/root_n)'''

np.random.seed(42)
sample = np.random.normal(33, 7, 49)

# H0: mu = 30, H1: mu > 30
mu_0 = 30
sigma = 7
n = 49
alpha = 0.05

x_bar = np.mean(sample)
se = sigma / np.sqrt(n)
z_stat = (x_bar - mu_0) / se
z_critical = 1.645

print(f"Sample mean = {x_bar:.2f}")
print(f"Z statistic = {z_stat:.2f}")
print(f"Z critical  = {z_critical}")

if z_stat > z_critical:
    print("Decision: Reject H0 — delivery time significantly greater than 30 mins")
else:
    print("Decision: Fail to reject H0")
