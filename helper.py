from typing import NewType
import jax

def astuple(thing):
    return tuple(sorted(thing))

class Block(list):
    def containsall(self, thing) -> bool:
        return all([t in self for t in thing])
    def containsany(self, thing) -> bool:
        return any([t in self for t in thing])
    def add(self, thing) -> None:
        self.append(thing)
    def as_set(self) -> frozenset:
        return frozenset(self)
    def is_invalid(self) -> bool:
        return len(self) > len(self.as_set())

RNGType = jax.typing.ArrayLike