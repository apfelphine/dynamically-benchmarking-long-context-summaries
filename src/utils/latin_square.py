from typing import Any, List


def balanced_row(n: int, row_idx: int) -> List[Any]:
    result = []
    j, h = 0, 0
    array = list(range(n))

    for i in range(n):
        if i < 2 or i % 2 != 0:
            val = j
            j += 1
        else:
            val = n - h - 1
            h += 1

        idx = (val + row_idx) % n
        result.append(array[idx])

    # For odd-sized squares, reverse every second row to ensure balancing
    if n % 2 != 0 and row_idx % 2 != 0:
        result.reverse()

    return result


def balanced_latin_square(items: List[Any]):
    latin_model = []
    n = len(items)

    rows = n if n % 2 == 0 else n * 2

    for i in range(rows):
        index_row = balanced_row(n, i)
        latin_model.append([items[idx] for idx in index_row])
    return latin_model
