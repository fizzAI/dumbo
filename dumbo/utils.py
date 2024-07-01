from enum import Enum

class PreservingEnum(Enum):
    """Enum that preserves missing values"""

    def __init__(self, wild):
        super().__init__(self, wild)
        self._value = self._value_

    @property
    def value(self):
        return self.__getattribute__('_value')
    
    @value.setter
    def value(self, value):
        self.__setattr__('_value', value)

    @classmethod
    def _missing_(cls, value):
        res = cls(0)
        res.value = value
        return res