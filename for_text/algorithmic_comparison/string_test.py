import pymeleon as pym

def prefix_check(x) -> bool:
    return isinstance(x, str) and x == "banana"

def postfix_check(x) -> bool:
    return isinstance(x, str) and x == "apple"
    
def add_prefix(x: str, y: str) -> str:
    return f"{y}_{x}"

def add_postfix(x: str, y: str) -> str:
    return f"{x}_{y}"
 
def combine_str(x: str, y: str) -> str:
    return f"{x}_combined_{y}"

def split_str(x: str) -> list:
    return [c for c in x]

viewer = pym.DSL(
    pym.Predicate("prefix", prefix_check),
    pym.Predicate("postfix", postfix_check),
    pym.Rule(pym.parse({"a": str, "b": "prefix"}),
             pym.parse("add_prefix(a, b)", {"add_prefix": "prefixed"})),
    pym.Rule(pym.parse({"a": "prefixed", "b": "postfix"}),
             pym.parse("add_postfix(a, b)", {"add_postfix": "postfixed"})),
    pym.Rule(pym.parse({"a": "postfixed", "b": "postfixed"}),
             pym.parse("combine_str(a, b)", {"combine_str": "combined"})),
    pym.Rule(pym.parse({"a": "combined"}),
             pym.parse("split_str(a)", {"split_str": "split"})),
    name="string_test"
) >> pym.GeneticViewer(ext=[add_prefix, add_postfix, combine_str, split_str], 
                       use_pretrained=True,
                       hyperparams={"num_epochs": 10000}, 
                       device_str="cuda")

def ex_1(a: str, b: str):
    """
    Apply prefix
    """
    return viewer(a, b) >> pym.parse({"a": "prefixed"})

def ex_2(a: str, b: str, c: str):
    """
    Apply 2 prefixes
    """
    return viewer(a, b, c) >> pym.parse({"a": "prefixed", "b": "prefixed"})

def ex_3(a: str, b: str, c: str, d: str):
    """
    Apply 2 prefixes and 2 postfixes
    """
    return viewer(a, b, c, d) >> pym.parse({"a": "postfixed", "b": "postfixed"})

def ex_4(a: str, b: str, c: str, d: str):
    """
    Apply 2 prefixes, 2 postfixes and combine
    """
    return viewer(a, b, c, d) >> pym.parse({"a": "combined"})

def ex_5(a: str, b: str, c: str, d: str):
    """
    Apply 2 prefixes, 2 postfixes, combine and split
    """
    return viewer(a, b, c, d) >> pym.parse({"a": "split"})

x = "hello"
y = "world"
prefix = "banana"
postfix = "apple"
print(ex_1(x, prefix),
      ex_2(x, y, prefix),
      ex_3(x, y, prefix, postfix),
      ex_4(x, y, prefix, postfix),
      ex_5(x, y, prefix, postfix),
      sep="\n===============================\n")