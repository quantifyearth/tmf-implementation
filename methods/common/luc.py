from typing import Iterable

def luc_matching_columns(start_year: int) -> tuple[int, int, int]:
    luc0 = f'luc_{start_year}'
    luc5 = f'luc_{start_year - 5}'
    luc10 = f'luc_{start_year - 10}'
    return luc0,luc5,luc10


def luc_range(start_year: int, evaluation_year: int) -> Iterable[int]:
    return range(start_year - 10, evaluation_year + 1)

