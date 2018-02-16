# HPM573_SupportLib

This library includes classes to support the following analyses and operations.

-- EconEvalClasses.py --
1. identify strategies on the cost-effectiveness frontier
2. produce the table summarizing the result of the cost-effectiveness analysis 
3. produce cost-effectiveness planes,
4. calcualte the confidence intervals and projection intervals for ICER estmates

~RandomVariateGenerators.py~ 
To generate (thread-safe) realizations from various probability distributions. 

~SamplePathClass.py~
1. to record a sample path (prevalence) as the function of simulation time
2. to graph sample path(s) 

~StatisticalClasses.py~
Calculate sample mean, sample standard deviation, min, max, qth percentile, t-based confidence interval, empirical bootstrap confidence interval and percentile inteval for:
1. random variable X when realizations of X are available in batch 
2. random variable X when realizations of X become available over time
3. random variable X when X is a function of simulation time 
4. random variable Z = X-Y
5. random variable X = X/Y

~InOutFunctions.py~
1. write a list of lists to a csv file
2. read a matrix from a scv file and return in format of list of lists 
