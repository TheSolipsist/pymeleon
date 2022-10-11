import inspect
from pymeleon.dsl.parser import parse
from pymeleon.dsl.rule import Rule


def autorule(method):
    def name(class_obj):
        return class_obj.__name__
    signature = inspect.signature(method)

    for arg, parameter in signature.parameters.items():
        if parameter == inspect.Parameter.empty:
            raise Exception(f"autorule needs an annotation for argument {arg} of callable {name(method)}")
    if signature.return_annotation == inspect.Parameter.empty:
        raise Exception(f"autorule needs a return annotation or method {name(method)}")

    lhs = parse({arg: parameter.annotation for arg, parameter in signature.parameters.items()})
    rhs = parse(f"{name(method)}({','.join(arg for arg in signature.parameters)})",
                {name(method): signature.return_annotation})
    return Rule(lhs, rhs, ext={name(method): method})
