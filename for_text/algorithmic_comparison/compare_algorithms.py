"""
Create 4-5 DSLs and come up with 5 examples for each, ascending complexity
Compare algorithms with and without neural net for each DSL and measure mean
For each example write an explanation for what it does
Measure dataset generation time, DSL training time, mean inference time (100 iterations or sth, check how many of them find the solution)
Table, on the left problem name, 2 columns describing search time and results without neural net, 
"""
from for_text.algorithmic_comparison import pygrank_test, numpy_test, graph_test, string_test

modules = [pygrank_test,
           graph_test, 
           numpy_test,
           string_test]

n = 100
n_scen = 5
for mod in modules:
    print(mod.__name__)
    found = [0] * n_scen
    total_times = [0] * n_scen
    total_iterations = [0] * n_scen
    for _ in range(n):
        results = mod.test()
        for i, result in enumerate(results):
            if result is not False:
                found[i] += 1
                total_times[i] += result[1]
                total_iterations[i] += result[0][1]
    with open(f"results/experiments/CGP/{mod.__name__}", "w") as f:
        f.write(f"THIS IS {mod.__name__}\n\n")
        for i in range(n_scen):
            f.write(f"Scenario {i + 1}:\n")
            f.write(f"\tAccuracy: {found[i] / n}\n")
            f.write(f"\tIterations: {total_iterations[i] / n}\n")
            f.write(f"\tTime: {total_times[i] / n}\n")

print("DONE")