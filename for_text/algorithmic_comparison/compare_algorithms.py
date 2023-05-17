"""
Algorithmic comparison module for pymeleon
"""
import json
import pathlib

from for_text.algorithmic_comparison import (graph_test, numpy_test,
                                             pygrank_test, string_test)

modules = {
    mod: mod.__name__.split(".")[-1]
    for mod in (pygrank_test, graph_test, numpy_test, string_test)
}
n = 100
n_scen = 5
fitness_funcs = ("neural_random", "heuristic", "random")
json_dict = {
    "n": n,
}
for fitness_str in fitness_funcs:
    json_dict[fitness_str] = {}
    for mod, mod_name in modules.items():
        found = [0] * n_scen
        total_times = [0] * n_scen
        total_iterations = [0] * n_scen
        for j in range(n):
            print(f"\r{' ' * 100}\rTesting fitness function: {fitness_str}, module: {mod_name}, iter: {j + 1} / {n}", end="")
            results = mod.test(fitness_str)
            for i, result in enumerate(results):
                if result is not False:
                    found[i] += 1
                    total_times[i] += result[1]
                    total_iterations[i] += result[0][1]
        json_dict[fitness_str][mod_name] = {
            i + 1: {
                "accuracy": found[i] / n,
                "avg_iter_n": total_iterations[i] / found[i] if found[i] > 0 else "N/A",
                "avg_time": total_times[i] / found[i] if found[i] > 0 else "N/A",
            }
            for i in range(n_scen)
        }
print()
file_p = pathlib.Path(f"results/experiments.json")
file_p.parent.mkdir(parents=True, exist_ok=True)
with file_p.open(mode="w", encoding="utf-8") as json_file:
    json.dump(json_dict, json_file, indent=4)
