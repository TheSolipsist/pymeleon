import pymeleon as pym
import numpy as np
from pymeleon.utilities.util_funcs import timer
from pymeleon.viewer.genetic_viewer import ViewerError

def to_nparray(x: list) -> np.ndarray:
    return np.array(x)

def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(x * y)

def float_to_int(x: float) -> int:
    return int(x)

def copy_string(x: str, y: int) -> tuple:
    return (x,) * y

@timer
def ex_1(viewer, x: list):
    """
    Apply np.array transformation
    """
    return viewer(x) >> pym.parse(np.ndarray)

@timer
def ex_2(viewer, x: list, y: list):
    """
    Apply 2 np.array transformations
    """
    return viewer(x, y) >> pym.parse({"a": np.ndarray, "b": np.ndarray})

@timer
def ex_3(viewer, x: list, y: list):
    """
    Apply 2 np.array transformations and dot product
    """
    return viewer(x, y) >> pym.parse(float)

@timer
def ex_4(viewer, x: list, y: list):
    """
    Apply dot product transformation and and change to int
    """
    return viewer(x, y) >> pym.parse(int)

@timer
def ex_5(viewer, x: list, y: list, c: str):
    """
    Apply dot product transformation, change it to int and use that to get a list of int number of a string
    """
    return viewer(x, y, c) >> pym.parse(tuple)

def test_example(viewer, foo, *args):
    try:
        return foo(viewer, *args)
    except ViewerError:
        return False
    
def test(fitness_str: str):
    viewer = pym.DSL(
        pym.autorule(to_nparray),
        pym.autorule(dot_product),
        pym.autorule(float_to_int),
        pym.autorule(copy_string),
        name="numpy_test"
    ) >> pym.GeneticViewer(use_pretrained=True, hyperparams={"num_epochs": 10000}, device_str="cuda", fitness=fitness_str)
    x = [1, 2.3, 3]
    y = [4.2, 7.21, 12]
    c = "test"
    return (test_example(viewer, ex_1, x),
            test_example(viewer, ex_2, x, y),
            test_example(viewer, ex_3, x, y),
            test_example(viewer, ex_4, x, y),
            test_example(viewer, ex_5, x, y, c))
