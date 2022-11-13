
from enum import Enum

class TenseType(Enum):
    '''
    Reference: https://www.englisch-hilfen.de/en/grammar/tenses_table.pdf
    '''
    SimplePresent = 1
    PresentProgressive = 2

    SimplePast = 3
    PastProgressive = 4

    SimplePresentPerfect = 5
    PresentPerfectProgressive = 6

    SimplePastPerfect = 7
    PastPerfectProgressive = 8

    WillFuture = 9
    GoingToFuture = 10
    FutureProgressive = 11
    SimpleFuturePerfect = 12
    FuturePerfectProgressive = 13

    ConditionalSimple = 14
    ConditionalProgressive = 15
    ConditionalPerfect = 16
    ConditionalPerfectProgressive = 17

class SentenceType(Enum):
    Declarative = 1 # 陈述句
    Interrogative = 2 # 疑问句
    Imperative = 3 # 祈使句
    Exclamative = 4 # 感叹句

class InterrogativeType(Enum):
    YesNo = 1
    QuestionWord = 2
    Choice = 3
