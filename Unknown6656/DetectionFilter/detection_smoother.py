from enum import Enum, EnumMeta
from typing import TypeVar, Type
import time

import numpy as np

from .kalman_filter import SmoothingKalmanFilter


E = TypeVar('E', bound = Enum)


class Detection:
    '''Represents a detection as obtained from an object detector. A detection is described by the following properties:
    - center point (x, y)
    - width
    - height
    - dictionary of classes and their respective confidence/probability
    - detection confidence
    - detection timestamp (optional)
    '''

    center_x : float
    '''The detection's center point X coordinate (unit-agnostic).'''
    center_y : float
    '''The detection's center point Y coordinate (unit-agnostic).'''
    width : float
    '''The detection's width (unit-agnostic).'''
    height : float
    '''The detection's height (unit-agnostic).'''
    classes : dict[E, float]
    '''A dictionary representing a map of the object's possible classes and their respective confidence/probability.
    Each confidence/probability is a value in the inclusive interval 0..1.
    All values should (but are not required to) add to 1.
    '''
    confidence : float
    '''The detection confidence (a value in the inclusive interval 0..1).'''
    timestamp : float | None = None
    '''The detection timestamp (usually UNIX-time, in seconds).'''


    def __init__(self, center_x : float, center_y : float, width : float, height : float, classes : E | dict[E, float], confidence : float, timestamp : float | None = None):
        '''Creates a new Detection instance using the following parameters:
        - (float) center_x : The detection's center point X coordinate (unit-agnostic).
        - (float) center_y : The detection's center point Y coordinate (unit-agnostic).
        - (float) width : The detection's width (unit-agnostic).
        - (float) height : The detection's height (unit-agnostic).
        - (E) classes : A single enum value representing the detection's class.
        - (dict[E, float]) classes : A dictionary representing a map of the object's possible classes and their respective confidence/probability. Each confidence/probability is a value in the inclusive interval 0..1. All values should (but are not required to) add to 1.
        - (float) confidence : The detection confidence (a value in the inclusive interval 0..1).
        - (float) timestamp : The detection timestamp (usually UNIX-time, in seconds).
        '''
        if not isinstance(classes, dict):
            classes = { classes: 1. }

        for key in classes:
            if not issubclass(type(key), Enum):
                raise ValueError(f'The value "{key}" ({type(key)}) is not a valid value for a key of the parameter "classes", as it does not inherit from the class "{Enum}".')
            else:
                classes[key] = max(0., min(1., float(classes[key])))

        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.width = float(width)
        self.height = float(height)
        self.classes = classes
        self.confidence = max(0., min(1., float(confidence)))
        self.timestamp = None if timestamp is None else float(timestamp)

    def __repr__(self) -> str: return self.to_string()

    @property
    def object_class(self) -> E:
        '''Returns the object class with the highest confidence/probability.'''
        return max(self.classes, key = self.classes.get)

    @property
    def x_min(self) -> float: return self.center_x - self.width * .5

    @property
    def x_max(self) -> float: return self.center_x + self.width * .5

    @property
    def y_min(self) -> float: return self.center_y - self.height * .5

    @property
    def y_max(self) -> float: return self.center_y + self.height * .5

    def to_string(self) -> str: return f'{self.object_class.value} {self.center_x} {self.center_y} {self.width} {self.height} {self.confidence}'

    @staticmethod
    def from_string(string : str, enum_type : Type[E]) -> 'Detection':
        if not issubclass(enum_type, Enum):
            raise ValueError(f'The type "{enum_type}" is not a valid value for the parameter "enum_type", as it does not inherit from the class "{Enum}".')

        tokens = [s.strip() for s in string.split(' ')]

        if len(tokens) < 5:
            raise ValueError(f'The string "{string}" is not a valid Yolo representation of a detection.')
        elif (cls := next((x for x in enum_type if x.value == int(tokens[0])), None)) is None:
            raise ValueError(f'Unknown detection class "{tokens[0]}".')

        conf = 1. if len(tokens) <= 5 else float(tokens[5])

        return Detection(
            float(tokens[1]),
            float(tokens[2]),
            float(tokens[3]),
            float(tokens[4]),
            { cls: conf },
            conf
        )

    @staticmethod
    def from_file(path : str, enum_type : Type[E]) -> 'list[Detection]':
        def _inner():
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if len(line) > 8:
                        yield Detection.from_string(line, enum_type)
        return list(_inner())


class DetectionSmoother:
    '''A class which provides functionality for smoothing detections (as e.g. returned by YoloNet).'''
    _enum_type : type
    _filter : SmoothingKalmanFilter
    _dimension : int


    def __init__(self, enum_type : Type[E], q = 1.):
        if not issubclass(enum_type, Enum):
            raise ValueError(f'The type "{enum_type}" is not a valid value for the parameter "enum_type", as it does not inherit from the class "{Enum}".')

        self._enum_type = enum_type
        self._dimension = 4 + len(enum_type)
        self._filter = SmoothingKalmanFilter(self._dimension, q = q)

    def _detection_to_vectors(self, detection : Detection) -> tuple[np.ndarray, np.ndarray]:
        position = np.array([
            detection.center_x,
            detection.center_y,
            detection.width,
            detection.height
        ] + [detection.classes.get(cls, 0.) for cls in self._enum_type])
        accuracy = np.array([detection.confidence] * len(position))

        return position, accuracy

    def _vectors_to_detection(self, position : np.ndarray, accuracy : np.ndarray) -> Detection:
        x, y, w, h = position[:4]
        cls = {x: position[i + 4] for i, x in enumerate(self._enum_type)}

        return Detection(x, y, w, h, cls, np.average(accuracy))

    @property
    def minimum_accuracy(self) -> float:
        return self._filter.min_accuracy

    @minimum_accuracy.setter
    def minimum_accuracy(self, value : float):
        if value < 1e-5:
            raise ValueError(f'The filter\'s minimum accuracy must not be smaller than {1e-5}.')
        else:
            self._filter.min_accuracy = value

    def reset(self) -> None:
        self._filter.reset()

    def update(self, detection : Detection) -> None:
        if detection is None or not isinstance(detection, Detection):
            raise ValueError(f'No valid instance of "{Detection}" has been passed to the update function.')

        if isinstance(detection.classes, self._enum_type):
            detection.classes = {x: (1. if x == detection.classes else 0.) for x in self._enum_type}
        elif not(isinstance(detection.classes, dict) and all(isinstance(x, self._enum_type) and isinstance(detection.classes[x], float) for x in detection.classes)):
            raise ValueError(f'The field "classes" of the given {Detection} must be a value of the type "{self._enum_type}" or "dict[{self._enum_type}, {float}]".')

        timestamp = detection.timestamp or time.time()
        position, accuracy = self._detection_to_vectors(detection)

        self._filter.update(position, accuracy, timestamp)

        for i in range(4, self._dimension):
            pos = self._filter.position[i]
            vel = self._filter.velocity[i]

            (self._filter.position[i], self._filter.velocity[i]) =\
                (0., max(0., vel)) if pos < 0 else\
                (1., min(1., vel)) if pos > 1 else\
                (pos, vel)

    def estimate(self, timestamp : float | None = None) -> Detection:
        timestamp = timestamp or time.time()
        position = self._filter.estimate_position(timestamp)
        accuracy = self._filter.accuracy
        detection = self._vectors_to_detection(position, accuracy)
        detection.timestamp = timestamp

        return detection
