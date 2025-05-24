import numpy as np
import pandas as pd

EPSILON = 1e-9  # For floating point comparisons

# --- Simplex Algorithm Backend ---


def standardize_problem(obj_type, obj_coeffs, constraints_data, num_vars):
    """
    Converts the problem into standard form for the Simplex method.
    """
    obj_coeffs_std = np.array(obj_coeffs, dtype=float)
    if obj_type == "Minimize":
        obj_coeffs_std = -obj_coeffs_std
        was_minimized = True
    else:
        was_minimized = False

    # Initial variable names for decision variables
    var_names = [f"x{i+1}" for i in range(num_vars)]

    # Counters for new variables
    s_idx_counter = 0  # Slack
    e_idx_counter = 0  # Surplus
    a_idx_counter = 0  # Artificial

    # First pass: determine counts of slack, surplus, artificial vars
    # And prepare basic A_matrix rows for original vars
    num_slack = 0
    num_surplus = 0
    num_artificial = 0

    temp_A_rows = []
    temp_b_values = []
    temp_ops = []
    b_vector_list = []

    for constr in constraints_data:
        coeffs = np.array(constr['coeffs'], dtype=float)
        rhs = float(constr['rhs'])
        op = constr['op']

        if rhs < 0:
            coeffs = -coeffs
            rhs = -rhs
            if op == "<=":
                op = ">="
            elif op == ">=":
                op = "<="

        temp_A_rows.append(list(coeffs))  # Original variable coefficients
        temp_b_values.append(rhs)
        temp_ops.append(op)

        if op == "<=":
            num_slack += 1
        elif op == ">=":
            num_surplus += 1
            num_artificial += 1
        elif op == "=":
            num_artificial += 1

    # Extend objective function coefficients for all potential new variables (initially 0)
    obj_coeffs_std = np.concatenate(
        [obj_coeffs_std, np.zeros(num_slack + num_surplus + num_artificial)])

    # Second pass: build full A_matrix and var_names
    full_A_matrix_list = []

    current_s_col_offset = num_vars
    current_e_col_offset = num_vars + num_slack
    current_a_col_offset = num_vars + num_slack + num_surplus

    s_local_idx = 0
    e_local_idx = 0
    a_local_idx = 0

    basis_construction_details = []

    for i in range(len(temp_A_rows)):
        original_coeffs = temp_A_rows[i]
        op = temp_ops[i]

        row = list(original_coeffs) + [0.0] * \
            (num_slack + num_surplus + num_artificial)

        if op == "<=":
            s_idx_counter += 1
            var_name = f"s{s_idx_counter}"
            var_names.append(var_name)
            row[current_s_col_offset + s_local_idx] = 1.0
            basis_construction_details.append(
                {'type': 'slack', 'name': var_name, 'original_constraint_index': i})
            s_local_idx += 1
        elif op == ">=":
            e_idx_counter += 1
            surplus_var_name = f"e{e_idx_counter}"
            var_names.append(surplus_var_name)
            row[current_e_col_offset + e_local_idx] = -1.0
            e_local_idx += 1

            a_idx_counter += 1
            artificial_var_name = f"a{a_idx_counter}"
            var_names.append(artificial_var_name)
            row[current_a_col_offset + a_local_idx] = 1.0
            basis_construction_details.append(
                {'type': 'artificial', 'name': artificial_var_name, 'original_constraint_index': i})
            a_local_idx += 1
        elif op == "=":
            a_idx_counter += 1
            artificial_var_name = f"a{a_idx_counter}"
            var_names.append(artificial_var_name)
            row[current_a_col_offset + a_local_idx] = 1.0
            basis_construction_details.append(
                {'type': 'artificial', 'name': artificial_var_name, 'original_constraint_index': i})
            a_local_idx += 1

        full_A_matrix_list.append(row)
        b_vector_list.append(temp_b_values[i])

    return (
        np.array(full_A_matrix_list, dtype=float),
        np.array(b_vector_list, dtype=float),
        obj_coeffs_std,
        var_names,
        num_vars,  # original_num_vars
        num_slack,
        num_surplus,
        num_artificial,
        was_minimized,
        basis_construction_details
    )


def get_initial_basis(basis_construction_details, all_var_names):
    basis_indices = []
    basis_var_names = []

    for entry in basis_construction_details:
        basis_var_names.append(entry['name'])
        try:
            basis_indices.append(all_var_names.index(entry['name']))
        except ValueError:
            raise Exception(
                f"Variable {entry['name']} not found in all_var_names list for basis construction.")
    return basis_indices, basis_var_names


def create_tableau(A, b, obj_coeffs, var_names, basis_indices, phase_one=False, original_obj_coeffs_phase2=None, num_artificial_vars=0):
    num_constraints, num_total_vars_in_A = A.shape

    if len(obj_coeffs) != num_total_vars_in_A:
        if not phase_one and original_obj_coeffs_phase2 is not None and len(original_obj_coeffs_phase2) != num_total_vars_in_A:
            raise ValueError(
                f"Obj.coeffs length ({len(original_obj_coeffs_phase2)}) mismatch with A columns ({num_total_vars_in_A}) in Phase II.")
        elif phase_one and len(obj_coeffs) != num_total_vars_in_A:
            pass

    tableau_data = np.zeros((num_constraints + 1, num_total_vars_in_A + 1))
    tableau_data[:num_constraints, :num_total_vars_in_A] = A
    tableau_data[:num_constraints, -1] = b

    current_basis_var_names = [var_names[i] for i in basis_indices]

    if phase_one:
        cj_phase1 = np.zeros(num_total_vars_in_A)
        for i, name in enumerate(var_names):
            if name.startswith('a'):
                cj_phase1[i] = 1.0
        cb_phase1 = np.array([cj_phase1[idx] for idx in basis_indices])
        zj_phase1 = cb_phase1 @ A
        tableau_data[-1, :num_total_vars_in_A] = cj_phase1 - zj_phase1
        tableau_data[-1, -1] = -(cb_phase1 @ b)
    else:
        active_obj_coeffs = original_obj_coeffs_phase2 if original_obj_coeffs_phase2 is not None else obj_coeffs
        if len(active_obj_coeffs) != num_total_vars_in_A:
            raise ValueError(
                f"Phase II obj_coeffs length {len(active_obj_coeffs)} != A columns {num_total_vars_in_A}")
        cb = np.array([active_obj_coeffs[idx] for idx in basis_indices])
        zj = cb @ A
        tableau_data[-1, :num_total_vars_in_A] = active_obj_coeffs - zj
        tableau_data[-1, -1] = -(cb @ b)

    df_cols = var_names + ["RHS"]
    df_index = current_basis_var_names + ["Cj-Zj"]
    tableau_df = pd.DataFrame(tableau_data, columns=df_cols, index=df_index)
    return tableau_df


def find_pivot_column(tableau_df, phase_one=False):
    cj_zj_row = tableau_df.iloc[-1, :-1].values.astype(float)
    if phase_one:
        if np.all(cj_zj_row <= EPSILON):
            return -1
        pivot_col_idx = np.argmax(cj_zj_row)
        if cj_zj_row[pivot_col_idx] <= EPSILON:
            return -1
    else:
        if np.all(cj_zj_row <= EPSILON):
            return -1
        pivot_col_idx = np.argmax(cj_zj_row)
        if cj_zj_row[pivot_col_idx] <= EPSILON:
            return -1
    return pivot_col_idx


def find_pivot_row(tableau_df, pivot_col_idx):
    pivot_column_values = tableau_df.iloc[:-
                                          1, pivot_col_idx].values.astype(float)
    rhs_values = tableau_df.iloc[:-1, -1].values.astype(float)
    ratios = []
    valid_row_indices = []

    for i in range(len(pivot_column_values)):
        if pivot_column_values[i] > EPSILON:
            ratios.append(rhs_values[i] / pivot_column_values[i])
            valid_row_indices.append(i)

    if not valid_row_indices:
        return -1, "Unbounded: All pivot column elements are non-positive."

    min_ratio = float('inf')
    pivot_row_idx = -1
    for i, original_idx in enumerate(valid_row_indices):
        current_ratio = ratios[i]
        if current_ratio < min_ratio - EPSILON:
            min_ratio = current_ratio
            pivot_row_idx = original_idx
        elif abs(current_ratio - min_ratio) < EPSILON:
            if pivot_row_idx == -1 or original_idx < pivot_row_idx:
                pivot_row_idx = original_idx

    if pivot_row_idx == -1:
        return -1, "Error in ratio test, no valid pivot row found."

    ratio_strings = []
    for i in range(len(pivot_column_values)):
        desc_val = pivot_column_values[i]
        if desc_val > EPSILON:
            ratio_strings.append(
                f"{tableau_df.index[i]}: {rhs_values[i]:.2f}/{desc_val:.2f} = {rhs_values[i]/desc_val:.2f}")
        else:
            ratio_strings.append(f"{tableau_df.index[i]}: N/A (coeff ≤ 0)")
    min_ratio_text = f"Min Ratio = {min_ratio:.2f} (for row {tableau_df.index[pivot_row_idx]})"
    ratio_test_desc = "Ratio Test (RHS / Pivot Column Value):\n" + \
        "\n".join(ratio_strings) + "\n" + min_ratio_text
    return pivot_row_idx, ratio_test_desc


def perform_pivot_operation(tableau_df, pivot_row_idx, pivot_col_idx):
    pivot_col_idx_int = int(pivot_col_idx)
    new_tableau_df = tableau_df.copy()
    pivot_element = new_tableau_df.iloc[pivot_row_idx, pivot_col_idx_int]
    if abs(pivot_element) < EPSILON:
        raise ValueError(
            "Pivot element is zero, cannot perform pivot operation.")
    new_tableau_df.iloc[pivot_row_idx, :] /= pivot_element
    for i in range(new_tableau_df.shape[0]):
        if i != pivot_row_idx:
            factor = new_tableau_df.iloc[i, pivot_col_idx_int]
            new_tableau_df.iloc[i, :] -= factor * \
                new_tableau_df.iloc[pivot_row_idx, :]
    entering_var_name = new_tableau_df.columns[pivot_col_idx_int]
    new_basis_var_names = list(new_tableau_df.index)
    new_basis_var_names[pivot_row_idx] = entering_var_name
    new_tableau_df.index = new_basis_var_names
    return new_tableau_df


def format_tableau_html(df, iteration_str, description, pivot_r_idx=None, pivot_c_idx=None, entering_var=None, leaving_var=None):
    def style_pivot(data, pivot_r_idx, pivot_c_idx_as_int, entering_var, leaving_var):
        s = pd.DataFrame('', index=data.index, columns=data.columns)
        if pivot_r_idx is not None and pivot_c_idx_as_int is not None:
            if pivot_r_idx < len(data.index) - 1:
                for col_idx in range(len(data.columns) - 1):
                    s.iloc[pivot_r_idx, col_idx] = 'background-color: lightyellow;'
            if pivot_c_idx_as_int < len(data.columns) - 1:
                for row_idx in range(len(data.index) - 1):
                    current_style = s.iloc[row_idx, pivot_c_idx_as_int]
                    s.iloc[row_idx, pivot_c_idx_as_int] = (str(
                        current_style) + ';' if current_style else '') + 'background-color: lightblue;'
            s.iloc[pivot_r_idx, pivot_c_idx_as_int] = 'background-color: yellow; font-weight: bold; border: 1.5px solid red;'
        return s

    pivot_c_idx_int = int(pivot_c_idx) if pivot_c_idx is not None else None
    styled_df = df.style.set_table_attributes('class="dataframe table table-striped table-hover table-sm" border="1"') \
                        .format("{:.2f}")
    if pivot_r_idx is not None and pivot_c_idx_int is not None:
        styled_df = styled_df.apply(style_pivot, axis=None,
                                    pivot_r_idx=pivot_r_idx, pivot_c_idx_as_int=pivot_c_idx_int,
                                    entering_var=entering_var, leaving_var=leaving_var)
    html_table = styled_df.to_html(escape=False)
    header = f"<h3>{iteration_str}</h3>"
    if entering_var and leaving_var:
        header += f"<p><b>Entering Variable:</b> {entering_var}, <b>Leaving Variable:</b> {leaving_var}</p>"
    if description:
        header += f"<p>{description.replace(chr(10), '<br>')}</p>"
    return f"{header}{html_table}<hr>"


def solve_simplex_problem(obj_type, obj_coeffs_str, constraints_inputs, num_vars_val, num_constraints_val):
    steps_html = ""
    MAX_ITERATIONS = 50
    iteration_ph1 = 0  # Ensure it's defined for the `else` block after for-loop

    try:
        obj_coeffs_list = [float(x.strip()) for x in obj_coeffs_str.split(',')]
        if len(obj_coeffs_list) != num_vars_val:
            return f"<p style='color:red'>Error: Number of objective coefficients ({len(obj_coeffs_list)}) must match number of variables ({num_vars_val}).</p>", None
        parsed_constraints = []
        for i in range(num_constraints_val):
            coeffs_str = constraints_inputs[f"const_{i}_coeffs"]
            op = constraints_inputs[f"const_{i}_op"]
            rhs_str = constraints_inputs[f"const_{i}_rhs"]
            coeffs = [float(x.strip()) for x in coeffs_str.split(',')]
            if len(coeffs) != num_vars_val:
                return f"<p style='color:red'>Error: Constraint {i+1} coefficients ({len(coeffs)}) must match number of variables ({num_vars_val}).</p>", None
            parsed_constraints.append(
                {'coeffs': coeffs, 'op': op, 'rhs': float(rhs_str)})
    except ValueError as e:
        return f"<p style='color:red'>Error: Invalid numeric input for coefficients or RHS: {e}</p>", None
    except Exception as e:
        return f"<p style='color:red'>Error parsing inputs: {e}</p>", None

    A_std, b_std, obj_coeffs_for_tableau, var_names_std, orig_num_vars, num_s, num_sur, num_a, was_minimized, basis_constr_details = \
        standardize_problem(obj_type, obj_coeffs_list,
                            parsed_constraints, num_vars_val)
    steps_html += "<h2>Initial Problem Setup</h2>"
    steps_html += f"<p><b>Variables after standardization:</b> {', '.join(var_names_std)}</p><hr>"

    if num_a > 0:
        steps_html += "<h2>Phase I: Minimize Sum of Artificial Variables</h2>"
        phase_I_basis_indices, _ = get_initial_basis(
            basis_constr_details, var_names_std)
        tableau_df = create_tableau(A_std, b_std, obj_coeffs_for_tableau, var_names_std,
                                    phase_I_basis_indices, phase_one=True, num_artificial_vars=num_a)
        iteration = 0
        steps_html += format_tableau_html(
            tableau_df, f"Phase I - Iteration {iteration}", "Initial Phase I Tableau")
        for iteration_ph1 in range(1, MAX_ITERATIONS + 1):
            pivot_col_idx = find_pivot_column(tableau_df, phase_one=True)
            if pivot_col_idx == -1:
                phase_one_obj_val = - \
                    float(pd.to_numeric(
                        tableau_df.iloc[-1, -1], errors='coerce'))
                if abs(phase_one_obj_val) > EPSILON:
                    msg = f"Phase I ends. Sum of artificial variables W = {phase_one_obj_val:.4f} > 0. Problem is Infeasible."
                    steps_html += f"<p style='color:red;font-weight:bold;'>{msg}</p>"
                    return steps_html, {"status": "Infeasible", "message": msg}
                else:
                    steps_html += "<p style='color:green;font-weight:bold;'>Phase I ends. W = 0. Feasible solution found. Proceeding to Phase II.</p>"
                    current_basis_vars_final_ph1 = list(tableau_df.index[:-1])
                    current_basis_indices_final_ph1 = [var_names_std.index(
                        var) for var in current_basis_vars_final_ph1]
                    A_for_phase2 = tableau_df.iloc[:-1, :-1].values
                    b_for_phase2 = tableau_df.iloc[:-1, -1].values
                    tableau_df = create_tableau(A_for_phase2, b_for_phase2, obj_coeffs_for_tableau,
                                                list(
                                                    tableau_df.columns[:-1]), current_basis_indices_final_ph1,
                                                phase_one=False, original_obj_coeffs_phase2=obj_coeffs_for_tableau,
                                                num_artificial_vars=num_a)
                    break
            pivot_row_idx, ratio_desc = find_pivot_row(
                tableau_df, int(pivot_col_idx))
            if pivot_row_idx == -1:
                msg = "Phase I: Unbounded (or error in pivot row selection)."
                steps_html += f"<p style='color:orange;font-weight:bold;'>{msg}</p>"
                return steps_html, {"status": "Error in Phase I", "message": msg}
            entering_var = tableau_df.columns[int(pivot_col_idx)]
            leaving_var = tableau_df.index[pivot_row_idx]
            tableau_df = perform_pivot_operation(
                tableau_df, pivot_row_idx, pivot_col_idx)
            steps_html += format_tableau_html(tableau_df, f"Phase I - Iteration {iteration_ph1}", ratio_desc, pivot_row_idx, int(
                pivot_col_idx), entering_var, leaving_var)
        # Executed if the loop completed fully (no break, i.e. max iterations reached)
        else:
            if iteration_ph1 >= MAX_ITERATIONS:
                return steps_html + "<p style='color:red;font-weight:bold;'>Phase I: Exceeded maximum iterations.</p>", {"status": "Max Iterations in Phase I"}

    else:
        steps_html += "<h2>Simplex Method (No Phase I needed)</h2>"
        initial_basis_indices, _ = get_initial_basis(
            basis_constr_details, var_names_std)
        tableau_df = create_tableau(
            A_std, b_std, obj_coeffs_for_tableau, var_names_std, initial_basis_indices, phase_one=False)

    phase_name = "Phase II" if num_a > 0 else "Simplex Solution"
    steps_html += f"<h2>{phase_name}</h2>"
    iteration = 0
    steps_html += format_tableau_html(
        tableau_df, f"{phase_name} - Iteration {iteration}", f"Initial Tableau for {phase_name}")
    for iteration_ph2 in range(1, MAX_ITERATIONS + 1):
        pivot_col_idx = find_pivot_column(tableau_df, phase_one=False)
        if pivot_col_idx == -1:
            obj_val_final = - \
                float(pd.to_numeric(tableau_df.iloc[-1, -1], errors='coerce'))
            if was_minimized:
                obj_val_final = -obj_val_final
            solution = {"status": "Optimal",
                        "value": obj_val_final, "variables": {}}
            for i in range(orig_num_vars):
                var_name_orig = f"x{i+1}"
                if var_name_orig in tableau_df.index:
                    solution["variables"][var_name_orig] = float(
                        float(pd.to_numeric(tableau_df.loc[var_name_orig, "RHS"], errors='coerce')))
                else:
                    solution["variables"][var_name_orig] = 0.0
            steps_html += f"<p style='color:green;font-weight:bold;'>{phase_name} ends. Optimal solution found.</p>"
            return steps_html, solution
        pivot_row_idx, ratio_desc = find_pivot_row(
            tableau_df, int(pivot_col_idx))
        if pivot_row_idx == -1:
            msg = f"{phase_name} ends. Problem is Unbounded. ({ratio_desc})"
            steps_html += f"<p style='color:orange;font-weight:bold;'>{msg}</p>"
            return steps_html, {"status": "Unbounded", "message": msg}
        entering_var = tableau_df.columns[int(pivot_col_idx)]
        leaving_var = tableau_df.index[pivot_row_idx]
        tableau_df = perform_pivot_operation(
            tableau_df, pivot_row_idx, pivot_col_idx)
        steps_html += format_tableau_html(tableau_df, f"{phase_name} - Iteration {iteration_ph2}",
                                          ratio_desc, pivot_row_idx, int(pivot_col_idx), entering_var, leaving_var)
    return steps_html + f"<p style='color:red;font-weight:bold;'>{phase_name}: Exceeded maximum iterations.</p>", {"status": f"Max Iterations in {phase_name}"}
