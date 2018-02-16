# Support Library

This library includes classes to support the following analyses and operations.

** EconEvalClasses.py **
1. To identify strategies that are on the cost-effectiveness frontier
2. To produce the table summarizing the result of the cost-effectiveness analysis 
3. To produce cost-effectiveness planes,
4. To calculate confidence intervals and projection intervals for ICER estimates

** RandomVariateGenerators.py **
1. To generate (thread-safe) realizations from various probability distributions 

** SamplePathClass.py **
1. To record a sample path (prevalence) as the function of simulation time
2. To graph sample path(s) 

** StatisticalClasses.py ** \
To calculate sample mean, sample standard deviation, min, max, q-th percentile, t-based confidence interval, empirical bootstrap confidence interval and percentile inteval for:
1. random variable X when realizations of X are available in batch 
2. random variable X when realizations of X become available over time
3. random variable X(t) when X(t) is a continuous function of simulation time 
4. random variable Z = X-Y
5. random variable X = X/Y

** InOutFunctions.py **
1. To write a list of lists to a csv file
2. To read a matrix from a scv file and return in format of list of lists 
