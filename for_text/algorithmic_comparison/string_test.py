import pymeleon as pym

def add_prefix(x: str) -> str:
    return f"banana_{x}"

def add_postfix(x: str) -> str:
    return f"{x}_apple"
 
def combine_str(x: str, y: str) -> str:
    return f"{x}_combined_{y}"

def split_str(x: str) -> list:
    return [c for c in x]

viewer = pym.DSL(
    pym.Rule(pym.parse(str),
             pym.parse("add_prefix(_)", {"add_prefix": "prefixed"})),
    pym.Rule(pym.parse({"a": "prefixed"}),
             pym.parse("add_postfix(a)", {"add_postfix": "postfixed"})),
    pym.Rule(pym.parse({"a": "postfixed", "b": "postfixed"}),
             pym.parse("combine_str(a, b)", {"combine_str": "combined"})),
    pym.Rule(pym.parse({"a": "combined"}),
             pym.parse("split_str(a)", {"split_str": "split"})),
    name="string_test"
) >> pym.GeneticViewer(ext=[add_prefix, add_postfix, combine_str, split_str], 
                       use_pretrained=True,
                       hyperparams={"num_epochs": 10000}, 
                       device_str="cuda")

def ex_1(x: str):
    """
    Apply prefix
    """
    return viewer(x) >> pym.parse({"a": "prefixed"})

def ex_2(x: str, y: str):
    """
    Apply 2 prefixes
    """
    return viewer(x, y) >> pym.parse({"a": "prefixed", "b": "prefixed"})

def ex_3(x: str, y: str):
    """
    Apply 2 prefixes and 2 postfixes
    """
    return viewer(x, y) >> pym.parse({"a": "postfixed", "b": "postfixed"})

def ex_4(x: str, y: str):
    """
    Apply 2 prefixes, 2 postfixes and combine
    """
    return viewer(x, y) >> pym.parse({"a": "combined"})

def ex_5(x: str, y: str):
    """
    Apply 2 prefixes, 2 postfixes, combine and split
    """
    return viewer(x, y) >> pym.parse({"a": "split"})

x = "hello"
y = "world"
print(ex_1(x),
      ex_2(x, y),
      ex_3(x, y),
      ex_4(x, y),
      ex_5(x, y),
      sep="\n===============================\n")