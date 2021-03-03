import pprint
from typing import Any
pprint.pprint(globals()['__builtins__'])


class A:
    def __init__(self):
        self.__setattr__('value', 0)

    def __add__(self, other):
        return self.value

    def foo(self):
        return self.value

    def __setattr__(self, name: str, value: Any) -> None:
        print(self)


a = A()
# (a + 2) = 1
