from typing import Any, TypeVar

FunctionType = TypeVar('FunctionType')
ReturnType = TypeVar('ReturnType')

class Event:
    def __init__(self, name:str=None):
        self.name = name
        self.subscribers = []
    def subscribe(self, object:Any) -> None:
        self.subscribers.append(object)
    def publish(self, *args, **kwargs) -> None:
        for i in range(len(self.subscribers)):
            self.subscribers[i](*args, **kwargs)
    def subscriber(self, function:FunctionType) -> FunctionType:
        self.subscribers.append(function)
        return function
    def __str__(self) -> str:
        return repr(self)
    def __repr__(self) -> str:
        return f"Event(subscribers={len(self.subscribers)})"
