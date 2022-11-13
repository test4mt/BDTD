

from enum import Enum
from http.client import CONFLICT

class MutationResult(Enum):
    MUTATED = "MUTATED"
    DUMP = "DUMP"
    CONFLICT = "CONFLICT"
    NER = "NER"