import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib import pyplot as plt


def calculate_y(x):
    return np.where(x == 0, 1, np.sin(x) / x)


def calculate_z(x, y):
    return (x - y) * np.sin(x + y)


def create_fuzzy_variables(fuzz_type, min_val, max_val, segments):
    domain = np.linspace(min_val, max_val, 500)
    mx = ctrl.Antecedent(domain, "mx")
    my = ctrl.Antecedent(domain, "my")
    mf = ctrl.Consequent(domain, "mf")
    midpoints = np.linspace(min_val, max_val, segments)
    step_size = midpoints[1] - midpoints[0]

    setup_memberships(mx, midpoints, fuzz_type, max_val - min_val, step_size)
    setup_memberships(my, midpoints, fuzz_type, max_val - min_val, step_size)

    if fuzz_type == "Trapezoidal" or fuzz_type == "Gaussian":
        output_midpoints = np.linspace(min_val, max_val, int(segments * 1.5))
        output_step = output_midpoints[1] - output_midpoints[0]
        setup_memberships(mf, output_midpoints, fuzz_type, max_val - min_val, output_step)
    else:
        mf.automf(names=[f"mf{i}" for i in range(1, 10)])

    return mx, my, mf

def run_fuzzy_simulation(fuzz_type, min_val, max_val, segments, title, use_diag_rules=False):
    mx, my, mf = create_fuzzy_variables(fuzz_type, min_val, max_val, segments)
    rules = define_diag_rules(mx, my, mf) if use_diag_rules else define_fuzzy_rules(mx, my, mf)

    fuzzy_system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(fuzzy_system)

    input_vals = np.linspace(min_val, max_val, segments * 10)
    true_y = calculate_y(input_vals)
    true_z = calculate_z(input_vals, true_y)
    pred_y = np.zeros_like(input_vals)
    pred_z = np.zeros_like(input_vals)

    for i in range(len(input_vals)):
        sim.input["mx"] = input_vals[i]
        sim.input["my"] = true_y[i]
        sim.compute()
        try:
            pred_y[i] = sim.output["mf"]
            pred_z[i] = calculate_z(input_vals[i], pred_y[i])
        except KeyError:
            print(f"Warning: No output 'mf' for inputs mx = {input_vals[i]} and my = {true_y[i]}")
            pred_y[i] = np.nan
            pred_z[i] = np.nan

    valid_idxs = ~np.isnan(pred_z)
    error = calculate_error(true_z[valid_idxs], pred_z[valid_idxs])

    plt.figure()
    plt.plot(input_vals[valid_idxs], true_z[valid_idxs], label="True Values", color="blue")
    plt.plot(input_vals[valid_idxs], pred_z[valid_idxs], label="Fuzzy Model", color="red", linestyle="--")
    plt.title(f"{title}: Mean Error: {error:.2%}")
    plt.xlabel("Input (x)")
    plt.ylabel("Output (z)")
    plt.legend()
    plt.show()

    print(f"{title} Error: {error:.2%}")

def setup_memberships(fuzzy_var, midpoints, mf_type, range_size, interval):
    if mf_type == "Trapezoidal":
        for i, (a, b, c, d) in enumerate(
                zip(midpoints - interval, midpoints - interval / 4, midpoints + interval / 4, midpoints + interval)
        ):
            fuzzy_var[f"{fuzzy_var.label}{i + 1}"] = fuzz.trapmf(fuzzy_var.universe, [a, b, c, d])
    elif mf_type == "Triangular":
        fuzzy_var.automf(names=[f"{fuzzy_var.label}{i}" for i in range(1, 7)])
    elif mf_type == "Gaussian":
        for i, center in enumerate(midpoints):
            fuzzy_var[f"{fuzzy_var.label}{i + 1}"] = fuzz.gaussmf(fuzzy_var.universe, center, range_size / 10)


def define_fuzzy_rules(mx, my, mf):
    rules_list = []
    for i in range(1, 7):
        for j in range(1, 7):
            out_index = determine_output_idx(i, j)
            rule = ctrl.Rule(mx[f"mx{i}"] & my[f"my{j}"], mf[f"mf{out_index}"])
            rule.label = f'Rule {(i * 6 + j) - 6}: If mx{i} and my{j} then mf{out_index}'
            print(rule.label)
            rules_list.append(rule)
    return rules_list


def determine_output_idx(x, y):
    if x == 1 or y == 1:
        return 1
    if x == 2:
        return 1 if y < 3 else 2 if y < 6 else 3
    if x == 3:
        return 1 if y < 3 else 2 if y == 3 else 3 if y < 6 else 4
    if x == 4:
        return 1 if y < 3 else 3 if y == 3 else 4 if y == 4 else 5 if y == 5 else 6
    if x == 5:
        return 1 if y == 1 else 2 if y == 2 else y + 1
    if x == 6:
        return 1 if y == 1 else y + 1 if y < 4 else y + 2 if y < 6 else y + 3
    return 1


def define_diag_rules(mx, my, mf):
    rules_list = []
    for i in range(1, 7):
        out_index = diagonal_output_idx(i)
        rule = ctrl.Rule(mx[f'mx{i}'] & my[f'my{i}'], mf[f'mf{out_index}'])
        rule.label = f'Rule {i}: If mx{i} and my{i} then mf{out_index}'
        print(rule.label)
        rules_list.append(rule)
    return rules_list

def calculate_error(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))

def diagonal_output_idx(x):
    return {1: 1, 2: 1, 3: 2, 4: 4, 5: 6, 6: 9}.get(x, 1)


for fuzz_type in ["Triangular", "Trapezoidal", "Gaussian"]:
    run_fuzzy_simulation(fuzz_type, 0.5, 1.5, 6, fuzz_type)
    if fuzz_type == "Gaussian":
        run_fuzzy_simulation(fuzz_type, 0.5, 1.5, 6, fuzz_type, use_diag_rules=True)
