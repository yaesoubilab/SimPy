import time


def fib(n):
    if n < 2:
        return n
    return fib(n-2) + fib(n-1)


def fib_cached(n, cache={}):
    if n < 2:
        return n
    try:
        val = cache[n]
    except KeyError:
        val = fib_cached(n-2, cache) + fib_cached(n-1, cache)
        cache[n] = val
    return val


N = 40
start = time.time()
v = fib(N)
end = time.time()
print(v, end - start)

start = time.time()
v = fib_cached(N)
end = time.time()
print(v, end - start)

