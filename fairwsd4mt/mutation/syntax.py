

from typing import List, Union

def find_child_index(original_deps: List[str], original_head, index: int, type: Union[str, List]) -> int:
    types = []
    if isinstance(type, str):
        types.append(type)
    else:
        types = type
    for i in range(len(original_deps)):
        if original_head[i].i == index and original_deps[i] in types:
            return i
    return -1

def find_children_indices(original_deps: List[str], original_head, index: int, type: Union[str, List]) -> List[int]:
    types = []
    if isinstance(type, str):
        types.append(type)
    else:
        types = type
    
    ret = []
    for i in range(len(original_deps)):
        if original_head[i].i == index and original_deps[i] in types:
            ret.append(i)
    return ret

def recover_word(cleaned_word: str, original_word: str) -> str:
    '''
    cleaned_word: good
    original_word: Goods<Space>
    output: Good<Space>
    '''
    ret = cleaned_word
    if original_word[0].isupper():
        ret = ret.capitalize()
    
    if original_word.endswith(" ") and not ret.endswith(" "):
        ret = ret + " "
    
    return ret