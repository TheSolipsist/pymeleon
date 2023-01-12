# Data Viewers

1. [Viewers](#viewers)
2. [Data blobs](#data-blobs)
3. [Transform blobs](#transform-blobs)

## Viewers
Viewers declare strategies for searching for data
transformations based on DSL rules and constraints.
For the time being, the best viewer supported by 
`pymeleon` is the `GeneticViewer`, which employs
genetic algorithms with neurally learned fitness
functions to guide the evolutionary process of
potentical solutions. Instantiating such a viewer
can be done with the following snippet:

```python
from pymeleon import GeneticViewer

viewer = GeneticViewer(use_pretrained=True, hyperparams={"num_epochs": 1000}, device_str="cuda")
```

The `use_pretrained` argument of this particular viewer
sets whether it loads pretrained fitness functions 
for the particular DSL based on its name. This 
parameter is *True* by default.
This viewer's constructor also takes 
hyperparameters for learning a neural fitness function
as well as the device (*cpu* or *cuda*) on which `torch` models will be run.

Attaching a DSL on rhe viewer can be done with the
shift operator per:

```python
viewer = dsl >> viewer
```

:bulb: Using this idiom, you can directly define a viewer 
tied to a DSL per:

```python
viewer = DSL(
    ... # add rules
).set_name("your_dsl_name") >> GeneticViewer(...)
```

## Data blobs

Blobs are unstructured collections of runtime data objects
that you want to transform to desired types. These
transformations can involve any combination allowed
by DSL rules. Blobs are declared by calling viewers, 
for instance per:

```python
blob = viewer(obj1, obj2, ...)
```

## Transform blobs

Given that you have constructed a blob from a viewer,
you can finally convert them to desired expressions.
This can also be done with the shift operator:

```python
desired_outcome = blob >> expression
```

This final step calls the viewer's strategy of synthesizing
the desired outcome by appropriately 
applying DSL transformation.

:warning: This operation may fail, so always write tests
to check that common use cases are handled correctly.
Using the same pretrained fitness functions will always
succeed on the same synthesis task, so prefer
[deploying pretrained fitness functions](deploy.md).
