from pymeleon import DSL, GeneticViewer, autorule, parse, Predicate
import numpy as np


def list2array(x: list) -> np.ndarray:
    return np.array(x)


def scale(x: np.ndarray, factor: "positive") -> np.ndarray:
    return x*factor


viewer = DSL(
    Predicate("positive", lambda x: isinstance(x, float) and x > 0),
    autorule(list2array),
    autorule(scale)
).set_name("demo_dsl") >> GeneticViewer()

print(viewer([1, 2, 3], 3.) >> np.ndarray)
