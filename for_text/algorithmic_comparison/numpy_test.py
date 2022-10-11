import pymeleon as pym
import numpy as np

def to_nparray(x: list) -> np.ndarray:
    return np.array(x)

def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum(x * y)

def float_to_int(x: float) -> int:
    return int(x)

def many_strings(x: str, y: int) -> tuple:
    return (x,) * y

viewer = pym.DSL(
    pym.autorule(to_nparray),
    pym.autorule(dot_product),
    pym.autorule(float_to_int),
    pym.autorule(many_strings),
    name="numpy_test"
) >> pym.GeneticViewer(use_pretrained=True, hyperparams={"num_epochs": 1000}, device_str="cuda")

def ex_1(x: list):
    """
    Apply np.array transformation
    """
    return viewer(x) >> pym.parse(np.ndarray)

def ex_2(x: list, y: list):
    """
    Apply 2 np.array transformations
    """
    return viewer(x, y) >> pym.parse({"a": np.ndarray, "b": np.ndarray})

def ex_3(x: list, y: list):
    """
    Apply 2 np.array transformations and dot product
    """
    return viewer(x, y) >> pym.parse(float)

def ex_4(x: list, y: list):
    """
    Apply dot product transformation and and change to int
    """
    return viewer(x, y) >> pym.parse(int)

def ex_5(x: list, y: list, c: str):
    """
    Apply dot product transformation, change it to int and use that to get a list of int number of a string
    """
    return viewer(x, y, c) >> pym.parse(tuple)

x = [1, 2.3, 3]
y = [4.2, 7.21, 12]
c = "test"
print(ex_1(x),
      ex_2(x, y),
      ex_3(x, y),
      ex_4(x, y),
      ex_5(x, y, c),
      sep="\n===============================\n")