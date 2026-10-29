def stair_ways(n: int):
    """Yield all the ways to climb a set of n stairs taking
    1 or 2 steps at a time.

    >>> list(stair_ways(0))
    [[]]
    >>> s_w = stair_ways(4)
    >>> sorted([next(s_w) for _ in range(5)])
    [[1, 1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 2]]
    >>> list(s_w) # Ensure you're not yielding extra
    []
    """
    if n == 0:
        yield []
    elif n == 1:
        yield [1]

    else:
        for path in stair_ways(n-1):
            yield [1] + path
        for path in stair_ways(n-2):
            yield [2] + path
s_w = stair_ways(4)
print(sorted([next(s_w) for _ in range(5)]))