

import copy
from typing import List, Optional, Tuple


class MutationSentence(object):
    def __init__(self, tokens: List[str]) -> None:
        self._tokens = copy.deepcopy(tokens)
        self._insert_tokens: List[Optional[List[str]]] = [None] * (len(tokens) + 1)
        self._delete_tokens: List[Optional[bool]] = [False] * len(tokens)
    
    def delete_token(self, index: int):
        self._delete_tokens[index] = True
    
    def insert_tokens(self, index: int, tokens: List[str]):
        if self._insert_tokens[index] is None:
            self._insert_tokens[index] = []
        self._insert_tokens[index].extend(tokens)
    
    def get_result(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        ret = []
        align = []

        for i in range(len(self._tokens)):
            if self._insert_tokens[i] is not None:
                ret.extend(self._insert_tokens[i])
            if not self._delete_tokens[i]:
                align.append((i, len(ret)))
                ret.append(self._tokens[i])

        return ret, align
    
    def __getitem__(self, key):
        return self._tokens[key]
  
    def __setitem__(self, key, newvalue):
       self._tokens[key] = newvalue
    
    def __len__(self) -> int:
        return len(self._tokens)
            