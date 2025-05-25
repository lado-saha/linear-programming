# simplex_logic.py
import numpy as np
import pandas as pd

EPSILON = 1e-9


def standardize_problem(obj_type, obj_coeffs, constraints_data, num_vars):
    # Convert obj_coeffs to a list of floats
    obj_coeffs = [float(c) for c in obj_coeffs]

    original_obj_coeffs_dict = {
        f"x{i+1}": obj_coeffs[i] for i in range(num_vars)}

    # Store if original problem was minimization
    was_minimized = (obj_type == "Minimize")

    # If minimizing, we maximize -Z. So negate original objective coefficients.
    # This negation happens when we define the Cj row for Phase II or for direct solve.

    var_names = [f"x{i+1}" for i in range(num_vars)]
    num_decision_vars = num_vars

    A_matrix_rows = []
    b_vector_vals = []

    s_idx = 0  # Slack variable counter (e.g., e1, e2 based on HEC doc)
    sur_idx = 0  # Surplus variable counter
    a_idx = 0  # Artificial variable counter

    basis_candidates = []  # To help determine initial basis
    artificial_var_names_list = []

    # Determine counts and names
    num_slack_vars = sum(1 for c in constraints_data if c['op'] == "<=")
    num_surplus_vars = sum(1 for c in constraints_data if c['op'] == ">=")
    num_artificial_vars_eq = sum(1 for c in constraints_data if c['op'] == "=")
    # Each >= constraint also gets an artificial var
    num_artificial_vars_ge = num_surplus_vars
    total_artificial_vars = num_artificial_vars_eq + num_artificial_vars_ge

    # Extend var_names: x_vars, then slack_vars (e_slack), then surplus_vars (e_surplus), then artificial_vars (a)
    current_s_idx = 0
    for _ in range(num_slack_vars):
        current_s_idx += 1
        var_names.append(f"e{current_s_idx}")  # HEC uses 'e' for slack too

    current_sur_idx = current_s_idx  # Continue numbering for 'e' variables
    for _ in range(num_surplus_vars):
        current_sur_idx += 1
        var_names.append(f"e{current_sur_idx}")  # Surplus vars

    current_a_idx = 0
    for _ in range(total_artificial_vars):
        current_a_idx += 1
        var_name = f"a{current_a_idx}"
        var_names.append(var_name)
        artificial_var_names_list.append(var_name)

    # Build A matrix and b vector
    # Pointers for adding 1s and -1s for slack/surplus/artificial
    s_ptr = num_decision_vars
    sur_ptr = num_decision_vars + num_slack_vars
    a_ptr = num_decision_vars + num_slack_vars + num_surplus_vars

    # Reset counters for unique naming during row construction
    s_idx_local = 0
    sur_idx_local = 0  # Used for surplus var 'e_k' naming/indexing
    a_idx_local = 0  # Used for artificial var 'a_k' naming/indexing

    # HEC notation uses 'e' for both slack and surplus, distinguished by context or later by +/- sign
    # We will name them e1, e2 for slacks, then e3, e4 for surplus if they appear in that order.
    # The var_names list already has this sequence.

    for i, constr in enumerate(constraints_data):
        coeffs = [float(c) for c in constr['coeffs']]
        rhs = float(constr['rhs'])
        op = constr['op']

        if rhs < 0:  # Ensure RHS is non-negative
            coeffs = [-c for c in coeffs]
            rhs = -rhs
            if op == "<=":
                op = ">="
            elif op == ">=":
                op = "<="

        # Initialize row with decision var coeffs and zeros for all other var types initially
        row = coeffs + [0.0] * (len(var_names) - num_decision_vars)

        if op == "<=":
            row[s_ptr + s_idx_local] = 1.0
            # Slack var enters basis
            basis_candidates.append(var_names[s_ptr + s_idx_local])
            s_idx_local += 1
        elif op == ">=":
            row[sur_ptr + sur_idx_local] = -1.0  # Subtract surplus
            # Add artificial variable
            row[a_ptr + a_idx_local] = 1.0
            # Artificial var enters basis
            basis_candidates.append(var_names[a_ptr + a_idx_local])
            sur_idx_local += 1
            a_idx_local += 1
        elif op == "=":
            row[a_ptr + a_idx_local] = 1.0
            # Artificial var enters basis
            basis_candidates.append(var_names[a_ptr + a_idx_local])
            a_idx_local += 1

        A_matrix_rows.append(row)
        b_vector_vals.append(rhs)

    A_std = np.array(A_matrix_rows, dtype=float)
    b_std = np.array(b_vector_vals, dtype=float)

    return A_std, b_std, var_names, basis_candidates, original_obj_coeffs_dict, \
        was_minimized, artificial_var_names_list


def create_hec_tableau(A_matrix, b_vector, var_names_all, basis_var_names_list,
                       obj_coeffs_dict_for_Cj,  # Cj values for the top row
                       cb_coeffs_dict_for_basis,  # Cb values for current basic vars
                       is_phase_one=False):

    num_constraints = A_matrix.shape[0]
    num_total_vars = A_matrix.shape[1]

    # Header for Cj values (top row of the tableau)
    cj_values = [obj_coeffs_dict_for_Cj.get(var, 0.0) for var in var_names_all]

    # Coef. Z (Cb) column for basic variables
    cb_values = [cb_coeffs_dict_for_basis.get(
        var, 0.0) for var in basis_var_names_list]

    # Main body: A_matrix and b_vector
    # Rows: Basic Variables | Cb | Variable Columns (A_matrix) | bi

    # Calculate Zj row
    zj_row_coeffs = np.zeros(num_total_vars)
    for i in range(num_total_vars):  # For each variable column
        col_sum = 0
        for r in range(num_constraints):  # For each basic variable row
            col_sum += cb_values[r] * A_matrix[r, i]
        zj_row_coeffs[i] = col_sum

    # Zj for RHS (current objective function value)
    zj_rhs = np.dot(cb_values, b_vector)

    # Calculate Cj - Zj row
    cj_minus_zj_row = np.array(cj_values) - zj_row_coeffs

    # Assemble the DataFrame
    # Data for the main part (Cb, Var.Base, A_matrix, bi)
    df_data = []
    for r in range(num_constraints):
        df_data.append([cb_values[r]] + [basis_var_names_list[r]
                                         ] + list(A_matrix[r, :]) + [b_vector[r]])

    # Add Zj row
    df_data.append([np.nan] + ["Zj"] + list(zj_row_coeffs) +
                   [zj_rhs])  # Cb for Zj is NaN or blank
    # Add Cj-Zj row
    # Cb & bi for Cj-Zj is NaN
    df_data.append([np.nan] + ["Cj-Zj"] + list(cj_minus_zj_row) + [np.nan])

    # Columns: Coef. Z | Var.base | x1 | x2 | ... | e1 | ... | a1 | ... | bi
    df_columns = ["Coef. Z (Cb)", "Var.base"] + var_names_all + ["bi"]

    tableau_df = pd.DataFrame(df_data, columns=df_columns)

    # For display, we'll prepend the Cj row separately in format_tableau_html
    # Store Cj values to be used by formatter
    tableau_df.attrs['Cj_header'] = [""] * 2 + \
        cj_values + [""]  # Align with df_columns

    return tableau_df


# HEC rule: Max Cj-Zj for Max problem
def find_pivot_column_hec(tableau_df, maximize_objective=True):
    cj_minus_zj_values = tableau_df[tableau_df["Var.base"] ==
                                    # Exclude Cb, Var.base, bi
                                    "Cj-Zj"].iloc[0, 2:-1].astype(float).values

    # Max problem (like Max -W in Phase I, or Max Z in Phase II)
    if maximize_objective:
        if np.all(cj_minus_zj_values <= EPSILON):
            return -1  # Optimal
        pivot_col_idx_in_cj_zj = np.argmax(cj_minus_zj_values)
        if cj_minus_zj_values[pivot_col_idx_in_cj_zj] <= EPSILON:
            return -1  # Optimal
    else:  # Minimize problem (not directly used if we always convert to Max -Z)
        if np.all(cj_minus_zj_values >= -EPSILON):
            return -1  # Optimal
        pivot_col_idx_in_cj_zj = np.argmin(cj_minus_zj_values)
        if cj_minus_zj_values[pivot_col_idx_in_cj_zj] >= -EPSILON:
            return -1  # Optimal

    # Index needs to be mapped back to full DataFrame column index (add 2 for 'Cb' and 'Var.base')
    return pivot_col_idx_in_cj_zj + 2


# pivot_col_df_idx is the actual column index in DataFrame
def find_pivot_row_hec(tableau_df, pivot_col_df_idx):
    pivot_column_name = tableau_df.columns[pivot_col_df_idx]

    # Exclude Zj and Cj-Zj rows from consideration for pivot row
    constraint_rows_df = tableau_df.iloc[:-2, :]

    pivot_column_coeffs = constraint_rows_df[pivot_column_name].astype(
        float).values
    bi_values = constraint_rows_df["bi"].astype(float).values

    min_ratio = float('inf')
    pivot_row_df_idx = -1  # Index within the full DataFrame tableau_df
    valid_ratios_exist = False

    ratio_details_list = []

    for i in range(len(pivot_column_coeffs)):
        coeff = pivot_column_coeffs[i]
        bi = bi_values[i]
        current_var_base = constraint_rows_df.iloc[i]["Var.base"]

        if coeff > EPSILON:
            valid_ratios_exist = True
            ratio = bi / coeff
            ratio_details_list.append(
                f"{current_var_base}: {bi:.2f}/{coeff:.2f} = {ratio:.2f}")
            if ratio < min_ratio - EPSILON:
                min_ratio = ratio
                # This is index relative to constraint_rows_df, so it's also the direct index in tableau_df
                pivot_row_df_idx = i
            elif abs(ratio - min_ratio) < EPSILON:  # Tie-breaking (simple: smallest index)
                if pivot_row_df_idx == -1 or i < pivot_row_df_idx:  # Check if this basic var has smaller index
                    pivot_row_df_idx = i
        else:
            ratio_details_list.append(f"{current_var_base}: N/A (coeff ≤ 0)")

    if not valid_ratios_exist:
        return -1, "Unbounded: All pivot column elements in constraint rows are non-positive."

    if pivot_row_df_idx == -1:  # Should be caught by valid_ratios_exist
        return -1, "Error: No valid pivot row found despite positive coefficients."

    ratio_desc = "Ratio Test (bi / Pivot Column Coeff):\n" + "\n".join(ratio_details_list) + \
                 f"\nMin Ratio = {min_ratio:.2f} (for row of {tableau_df.iloc[pivot_row_df_idx]['Var.base']})"

    return pivot_row_df_idx, ratio_desc


def perform_pivot_operation_hec(tableau_df, pivot_row_df_idx, pivot_col_df_idx):
    # We need to operate on the numerical part of the DataFrame
    # Let's extract A, b, Zj, Cj-Zj as numpy arrays for easier math

    # Get data for row operations (excluding Cb and Var.base columns for these ops)
    # Row indices: 0 to num_constraints-1 are constraints, num_constraints is Zj, num_constraints+1 is Cj-Zj
    # Column indices: 2 to num_total_vars+1 are var coeffs, num_total_vars+2 is bi

    # We will modify a copy of the numerical data and then reconstruct the DataFrame
    # This is safer than in-place modification of mixed-type DataFrame

    # Extract the A matrix, b vector, Zj row (coeffs only), Cj-Zj row (coeffs only)
    # Number of actual variables (x, e, a)
    num_vars_in_tableau = len(tableau_df.columns) - 3  # - Cb, Var.base, bi

    # Create a purely numerical matrix for operations:
    # Rows: constraints, Zj_coeffs, Cj_Zj_coeffs
    # Cols: var_coeffs, bi_values
    # Note: Zj and Cj-Zj rows are NOT directly transformed by pivot ops in HEC method.
    # They are RECALCULATED after the A matrix and b vector (and basis) are updated.

    # Constraint coeffs
    new_A_matrix = tableau_df.iloc[:-2, 2:-1].astype(float).values.copy()
    # bi for constraints
    new_b_vector = tableau_df.iloc[:-2, -1].astype(float).values.copy()

    # Adjust col_idx for A_matrix
    pivot_element = new_A_matrix[pivot_row_df_idx, pivot_col_df_idx - 2]

    if abs(pivot_element) < EPSILON:
        raise ValueError("Pivot element is zero.")

    # Normalize pivot row in new_A_matrix and new_b_vector
    new_A_matrix[pivot_row_df_idx, :] /= pivot_element
    new_b_vector[pivot_row_df_idx] /= pivot_element

    # Update other constraint rows
    for r in range(new_A_matrix.shape[0]):
        if r != pivot_row_df_idx:
            factor = new_A_matrix[r, pivot_col_df_idx - 2]
            new_A_matrix[r, :] -= factor * new_A_matrix[pivot_row_df_idx, :]
            new_b_vector[r] -= factor * new_b_vector[pivot_row_df_idx]

    # Update basis variable name list
    # Get current Var.base names
    new_basis_var_names = list(tableau_df.iloc[:-2, 1].values)
    entering_var_name = tableau_df.columns[pivot_col_df_idx]
    new_basis_var_names[pivot_row_df_idx] = entering_var_name

    # Reconstruct the tableau with updated A, b, and basis.
    # Zj and Cj-Zj will be recalculated by create_hec_tableau.
    # Need the Cj and Cb coefficient dictionaries for the current phase.

    # This requires passing Cj_dict and Cb_dict to this function or making them accessible.
    # For now, let's assume this function returns the new A, b, and basis,
    # and the main loop calls create_hec_tableau again.

    return new_A_matrix, new_b_vector, new_basis_var_names


def format_tableau_html_hec(tableau_df, iteration_str, description,
                            pivot_r_idx_df=None, pivot_c_idx_df=None,
                            entering_var=None, leaving_var=None):

    # Get the Cj header from DataFrame attributes
    cj_header_list = tableau_df.attrs.get('Cj_header', [])

    # Create the Cj row HTML
    cj_html_row = "<tr>" + \
        "".join(
            [f"<th>{str(c) if c is not None else ''}</th>" for c in cj_header_list]) + "</tr>"

    # Style the main DataFrame
    def style_pivot_hec(data_row_series, row_idx, pivot_r_idx_df, pivot_c_idx_df):
        # data_row_series is a Series (one row of the df)
        # row_idx is the integer index of this row in the original DataFrame
        # One style string per cell in the row
        styles = [''] * len(data_row_series)
        if pivot_r_idx_df is not None and pivot_c_idx_df is not None:
            # Style pivot row
            # Don't style Zj/Cj-Zj rows this way
            if row_idx == pivot_r_idx_df and row_idx < len(tableau_df) - 2:
                # Skip Cb, Var.base, bi for this generic highlight
                for c_idx in range(2, len(data_row_series) - 1):
                    styles[c_idx] = 'background-color: lightyellow;'

            # Style pivot column cells (for constraint rows only)
            if row_idx < len(tableau_df) - 2:  # Only for constraint rows
                if styles[pivot_c_idx_df]:
                    styles[pivot_c_idx_df] += ';'
                styles[pivot_c_idx_df] += 'background-color: lightblue;'

            # Style pivot element itself (must be in a constraint row)
            if row_idx == pivot_r_idx_df and row_idx < len(tableau_df) - 2:
                styles[pivot_c_idx_df] = 'background-color: yellow; font-weight: bold; border: 1.5px solid red;'
        return styles

    # We need to apply styling row by row because Styler.applymap is cell-wise,
    # and Styler.apply (axis=None) is table-wise which is complex for conditional per-cell.
    # A common approach is to build HTML manually or use Styler.set_td_classes if classes are predefined.
    # For dynamic styles like this, Styler.apply with axis=1 (row-wise) is feasible.

    html_parts = []
    # Header
    header_html = f"<h3>{iteration_str}</h3>"
    if entering_var and leaving_var:
        header_html += f"<p><b>Entering Variable:</b> {entering_var}, <b>Leaving Variable:</b> {leaving_var}</p>"
    if description:
        header_html += f"<p>{description.replace(chr(10), '<br>')}</p>"
    html_parts.append(header_html)

    # Table start
    html_parts.append(
        '<table class="dataframe table table-striped table-hover table-sm" border="1">')
    # Cj Header row
    html_parts.append(f"<thead>{cj_html_row}</thead>")

    # Main table body
    html_parts.append("<tbody>")

    for r_idx, row_series in tableau_df.iterrows():
        cell_styles = style_pivot_hec(
            row_series, r_idx, pivot_r_idx_df, pivot_c_idx_df)
        html_row = "<tr>"
        for c_idx, cell_val in enumerate(row_series):
            val_str = ""
            if pd.isna(cell_val):
                val_str = ""
            elif isinstance(cell_val, float):
                val_str = f"{cell_val:.2f}" if abs(
                    # Small numbers in sci notation
                    cell_val) > 1e-5 or cell_val == 0 else f"{cell_val:.2e}"
            else:
                val_str = str(cell_val)

            style_attr = f'style="{cell_styles[c_idx]}"' if cell_styles[c_idx] else ''
            html_row += f"<td {style_attr}>{val_str}</td>"
        html_row += "</tr>"
        html_parts.append(html_row)

    html_parts.append("</tbody></table><hr>")
    return "".join(html_parts)


def solve_simplex_problem_hec_style(obj_type_ui, obj_coeffs_str_ui, constraints_inputs_ui,
                                    num_vars_ui, num_constraints_ui):
    steps_html = ""
    MAX_ITERATIONS = 25  # Reduced for testing, can be increased

    try:
        obj_coeffs_list_ui = [float(x.strip())
                              for x in obj_coeffs_str_ui.split(',')]
        num_vars_val = int(num_vars_ui)
        num_constraints_val = int(num_constraints_ui)

        if len(obj_coeffs_list_ui) != num_vars_val:
            return f"<p style='color:red'>Error: Mismatch in objective coefficients and number of variables.</p>", None

        parsed_constraints_list = []
        for i in range(num_constraints_val):
            coeffs_str = constraints_inputs_ui[f"const_{i}_coeffs"]
            op = constraints_inputs_ui[f"const_{i}_op"]
            rhs_str = constraints_inputs_ui[f"const_{i}_rhs"]
            coeffs = [float(x.strip()) for x in coeffs_str.split(',')]
            if len(coeffs) != num_vars_val:
                return f"<p style='color:red'>Error: Constraint {i+1} coefficient mismatch.</p>", None
            parsed_constraints_list.append(
                {'coeffs': coeffs, 'op': op, 'rhs': float(rhs_str)})
    except Exception as e:
        return f"<p style='color:red'>Input Parsing Error: {e}</p>", None

    A, b, all_vars, initial_basis_vars, orig_obj_coeffs_map, was_min, artificial_vars = \
        standardize_problem(obj_type_ui, obj_coeffs_list_ui,
                            parsed_constraints_list, num_vars_val)

    steps_html += f"<h2>Initial Setup</h2><p>Variables: {', '.join(all_vars)}</p>"
    steps_html += f"<p>Artificial Vars: {', '.join(artificial_vars) if artificial_vars else 'None'}</p><hr>"

    current_A = A
    current_b = b
    current_basis_names = initial_basis_vars

    # --- Phase I ---
    if artificial_vars:
        steps_html += "<h2>Phase I: Minimize Sum of Artificial Variables (by Max -W)</h2>"

        # Cj for Phase I (Max -W): -1 for artificial vars, 0 for others
        cj_phase_I_dict = {
            var: (-1.0 if var in artificial_vars else 0.0) for var in all_vars}

        iteration_ph1 = 0
        current_W_val = None  # Initialize current_W_val to ensure it's always defined
        while iteration_ph1 < MAX_ITERATIONS:
            # Cb for Phase I: -1 if basic var is artificial, 0 otherwise
            cb_phase_I_dict = {
                var: (-1.0 if var in artificial_vars else 0.0) for var in current_basis_names}

            tableau = create_hec_tableau(current_A, current_b, all_vars, current_basis_names,
                                         cj_phase_I_dict, cb_phase_I_dict, is_phase_one=True)

            pivot_col_idx = find_pivot_column_hec(
                tableau, maximize_objective=True)  # Maximizing -W

            # Check W value: Zj value in 'bi' column for Zj row. This is -W. So W = -tableau.loc['Zj', 'bi']
            current_W_val = - \
                float(tableau[tableau["Var.base"] == "Zj"].iloc[0]["bi"])

            if pivot_col_idx == -1:  # Optimal for Phase I
                # Artificial vars still in solution with non-zero sum
                if abs(current_W_val) > EPSILON:
                    steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1} (Final)",
                                                          f"Phase I Optimal. W = {current_W_val:.4f}. Problem Infeasible.")
                    return steps_html, {"status": "Infeasible", "message": f"Artificial variables sum to {current_W_val:.4f}."}
                else:  # W = 0, feasible for original problem
                    steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1} (Final)",
                                                          f"Phase I Optimal. W = {current_W_val:.4f}. Feasible. Proceed to Phase II.")
                    # Prepare for Phase II: A, b, basis are from this final Phase I tableau
                    # Artificial variables that are non-basic can be conceptually dropped (their columns in Cj_Phase_II will be 0)
                    # If an artificial variable is basic at zero level, it needs careful handling (not common in simple cases)
                    break  # Exit Phase I loop

            pivot_row_idx, ratio_desc = find_pivot_row_hec(
                tableau, pivot_col_idx)

            if pivot_row_idx == -1:
                steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1}",
                                                      f"{ratio_desc}\nPhase I Unbounded (Error Condition). W = {current_W_val:.4f}")
                return steps_html, {"status": "Error", "message": "Phase I Unbounded (problem with artificials or setup)."}

            entering_var = tableau.columns[int(pivot_col_idx)]
            leaving_var = tableau.iloc[pivot_row_idx]["Var.base"]

            steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1}", ratio_desc,
                                                  pivot_r_idx_df=pivot_row_idx, pivot_c_idx_df=pivot_col_idx,
                                                  entering_var=entering_var, leaving_var=leaving_var)

            current_A, current_b, current_basis_names = \
                perform_pivot_operation_hec(
                    tableau, pivot_row_idx, pivot_col_idx)

            iteration_ph1 += 1
            if iteration_ph1 >= MAX_ITERATIONS:
                return steps_html + "<p style='color:red'>Phase I: Max iterations reached.</p>", {"status": "Max Iterations (Phase I)"}

        # Check if loop exited due to max iter and W still > 0
        if iteration_ph1 >= MAX_ITERATIONS and current_W_val is not None and abs(current_W_val) > EPSILON:
            return steps_html + "<p style='color:red'>Phase I: Max iterations reached and W > 0.</p>", {"status": "Max Iterations (Phase I), W > 0"}

    # --- Phase II ---
    steps_html += "<h2>Phase II: Solve Original Problem</h2>"

    # Cj for Phase II: from original_obj_coeffs_map. If minimizing, negate them for maximization.
    cj_phase_II_dict = {}
    for var in all_vars:
        original_coeff = orig_obj_coeffs_map.get(
            # Get original x1, x2 coeff
            var.split('_')[0] if '_' in var else var, 0.0)
        if var in artificial_vars:  # Artificial vars have 0 cost in Phase II objective
            cj_phase_II_dict[var] = 0.0
        else:
            cj_phase_II_dict[var] = - \
                original_coeff if was_min else original_coeff

    iteration_ph2 = 0
    while iteration_ph2 < MAX_ITERATIONS:
        # Cb for Phase II
        cb_phase_II_dict = {var_b: cj_phase_II_dict.get(
            var_b, 0.0) for var_b in current_basis_names}

        tableau = create_hec_tableau(current_A, current_b, all_vars, current_basis_names,
                                     cj_phase_II_dict, cb_phase_II_dict, is_phase_one=False)

        # Always maximizing (original or -Z_min)
        pivot_col_idx = find_pivot_column_hec(tableau, maximize_objective=True)

        if pivot_col_idx == -1:  # Optimal for Phase II
            final_obj_val_from_tableau = float(
                tableau[tableau["Var.base"] == "Zj"].iloc[0]["bi"])
            actual_obj_val = -final_obj_val_from_tableau if was_min else final_obj_val_from_tableau

            solution_vars = {}
            for i in range(num_vars_val):  # Only original decision variables
                var_name_orig = f"x{i+1}"
                if var_name_orig in current_basis_names:
                    row_idx = current_basis_names.index(var_name_orig)
                    solution_vars[var_name_orig] = float(current_b[row_idx])
                else:
                    solution_vars[var_name_orig] = 0.0

            steps_html += format_tableau_html_hec(tableau, f"Phase II - Iteration {iteration_ph2} (Optimal)",
                                                  f"Optimal Solution Found. Z = {actual_obj_val:.4f}")
            return steps_html, {"status": "Optimal", "value": actual_obj_val, "variables": solution_vars}

        pivot_row_idx, ratio_desc = find_pivot_row_hec(tableau, pivot_col_idx)

        if pivot_row_idx == -1:
            steps_html += format_tableau_html_hec(tableau, f"Phase II - Iteration {iteration_ph2}",
                                                  f"{ratio_desc}\nProblem Unbounded.")
            return steps_html, {"status": "Unbounded", "message": ratio_desc}

        entering_var = tableau.columns[int(pivot_col_idx)]
        leaving_var = tableau.iloc[pivot_row_idx]["Var.base"]

        steps_html += format_tableau_html_hec(tableau, f"Phase II - Iteration {iteration_ph2}", ratio_desc,
                                              pivot_r_idx_df=pivot_row_idx, pivot_c_idx_df=pivot_col_idx,
                                              entering_var=entering_var, leaving_var=leaving_var)

        current_A, current_b, current_basis_names = \
            perform_pivot_operation_hec(tableau, pivot_row_idx, pivot_col_idx)

        iteration_ph2 += 1
        if iteration_ph2 >= MAX_ITERATIONS:
            return steps_html + "<p style='color:red'>Phase II: Max iterations reached.</p>", {"status": "Max Iterations (Phase II)"}

    # Fallback if loops complete without returning (should be caught by max iter checks)
    return steps_html + "<p style='color:red'>Solver finished without a definitive result.</p>", {"status": "Unknown Error"}
