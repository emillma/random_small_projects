
# %%
edges = {1: [6, 8],
         2: [7, 9],
         3: [4, 8],
         4: [3, 9],
         5: [],
         6: [1, 7],
         7: [2, 6],
         8: [1, 3],
         9: [2, 4]}


def foo(current: int, taken: set = set()) -> int:
    total = 0
    for child in edges[current]:
        if child not in taken:
            taken.add(child)
            print(current, child, child not in taken)
            total += foo(child, taken)
    return total


print('total', foo(1))
# %%
