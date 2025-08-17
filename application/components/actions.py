class Action:
    pass

class ClickAction(Action):
    LEFT = 0
    RIGHT = 1
    MIDDLE = 2
    def __init__(self, type:int, position:tuple[float,float]):
        self.type = type
        self.position = position