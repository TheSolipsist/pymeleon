# pymeleon
Runtime type-driven synthesis 
based on domain specific language (DSL) transformations.

**Development:** Orestis Farmakis (https://github.com/TheSolipsist), Manios Krasanakis (maniospas@hotmail.com)<br>
**Dependencies:** `scipy`,`torch`,`networkx`,TODO

## Features
:rocket: Run code only once<br>
:arrow_upper_right: DSL definition<br>
:robot: Rule generation from type-hinted methods<br>
:writing_hand: Type constraints<br>

## Quickstart
To use the library, lets start from a couple of
methods will full type hints:

```python
import numpy as np

def list2array(x: list) -> np.ndarray:
    return np.array(x)

def scale(x: np.ndarray, factor: "positive") -> np.ndarray:
    return x*factor
```

The *"positive"* hint is not declared yet, 
but that's alright. We now convert these
methods to a `pymeleon` DSL, assign a name 
to it, and pass it to a `GeneticViewer` 
that will support type-driven synthesis.


```python
from pymeleon import DSL, GeneticViewer, Predicate, autorule

viewer = DSL(
    Predicate("positive", lambda x: isinstance(x, float) and x > 0),
    autorule(list2array),
    autorule(scale)
).set_name("demo_dsl") >> GeneticViewer()
```

The DSL definition comprises any number of custom predicates 
(i.e., that  are not python types and are passed as strings 
to typehints) and any number of DSL rules, which are transformations
with specific input and output types. Type combinations are
supported via [custom rule generation](docs/dsl.md) .

Finally, if you can provide your DSL viewer to
users of your code so that they can
transform collections of objects to a specific type.
They can do so by applying transformations on the objects
per:

```python
scale = viewer([1, 2, 3], 3.) >> np.ndarray
print(scale)
# [3. 6. 9.]
```

Again, more [complex output requirements](docs/dsl.md) 
can be declared.

## Documentation
[DSL Definition](docs/dsl.md)<br>
[Data Viewers](docs/viewers.md)
