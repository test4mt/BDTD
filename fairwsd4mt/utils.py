

from ast import Yield
from typing import Generator, Iterable, Iterator, List, TypeVar


T = TypeVar('T')
def list_chunked(lst: List[T], n: int) -> Generator[List[T], None, None]:
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

T = TypeVar('T')
def chunked(generator: Iterator[T], n: int) -> Generator[List[T], None, None]:
    """Yield successive n-sized chunks from iterable."""
    while True:
        ret = []
        try:
            for i in range(n):
                ret.append(next(generator))
            yield ret
        except StopIteration as e:
            yield ret
            break