# simplex_logic.py
import numpy as np
import pandas as pd

EPSILON = 1e-9


# ... (standardize_problem, get_initial_basis - assumed to be mostly okay, 
# but ensure var_names order is consistent: decision, then slack e_i, then surplus e_i, then artificial a_i)
# ... (standardize_problem function from previous correct version)
def standardize_problem(obj_type, obj_coeffs, constraints_data, num_vars):
    # Convert obj_coeffs to a list of floats
    obj_coeffs = [float(c) for c in obj_coeffs]
    
    original_obj_coeffs_dict = {f"x{i+1}": obj_coeffs[i] for i in range(num_vars)}
    was_minimized = (obj_type == "Minimize")

    var_names = [f"x{i+1}" for i in range(num_vars)]
    num_decision_vars = num_vars
    
    s_idx_counter = 0
    e_idx_counter = 0 # Surplus vars will also use 'e' but be distinguished by their role
    a_idx_counter = 0

    basis_candidates = [] 
    artificial_var_names_list = []

    # Temp lists for processing constraints
    temp_A_coeffs_list = []
    temp_b_values_list = []
    temp_ops_list = []
    constraint_types_detail = [] # To track what kind of vars are added for each original constraint

    for constr in constraints_data:
        coeffs = [float(c) for c in constr['coeffs']]
        rhs = float(constr['rhs'])
        op = constr['op']

        if rhs < 0:
            coeffs = [-c for c in coeffs]
            rhs = -rhs
            op = {"<=": ">=", ">=": "<=", "=": "="}[op]
        
        temp_A_coeffs_list.append(coeffs)
        temp_b_values_list.append(rhs)
        temp_ops_list.append(op)

        if op == "<=":
            s_idx_counter += 1
            constraint_types_detail.append({'type': 'slack', 'name': f"e{s_idx_counter}"})
        elif op == ">=":
            e_idx_counter += 1 # This 'e' is for surplus
            a_idx_counter += 1
            constraint_types_detail.append({'type': 'surplus_artificial', 
                                            'surplus_name': f"e{s_idx_counter + e_idx_counter}", # Continue e sequence
                                            'artificial_name': f"a{a_idx_counter}"})
        elif op == "=":
            a_idx_counter += 1
            constraint_types_detail.append({'type': 'equality_artificial', 
                                            'artificial_name': f"a{a_idx_counter}"})
    
    num_slack_vars = s_idx_counter
    num_surplus_vars = e_idx_counter
    num_artificial_vars = a_idx_counter

    # Construct final var_names: decision, then all 'e' (slacks then surpluses), then 'a'
    current_e_naming_idx = 0
    for i in range(num_slack_vars):
        current_e_naming_idx+=1
        var_names.append(f"e{current_e_naming_idx}")
    
    for i in range(num_surplus_vars):
        current_e_naming_idx+=1
        var_names.append(f"e{current_e_naming_idx}")

    for i in range(1, num_artificial_vars + 1):
        var_name = f"a{i}"
        var_names.append(var_name)
        artificial_var_names_list.append(var_name)

    # Build A matrix
    A_matrix_rows = []
    s_ptr_offset = num_decision_vars
    e_ptr_offset = num_decision_vars + num_slack_vars 
    a_ptr_offset = num_decision_vars + num_slack_vars + num_surplus_vars

    s_added_count = 0
    e_added_count = 0 # surplus
    a_added_count = 0

    for i in range(len(temp_A_coeffs_list)):
        row_coeffs = temp_A_coeffs_list[i]
        # Initialize row with decision var coeffs and zeros for all other var types initially
        full_row = list(row_coeffs) + [0.0] * (num_slack_vars + num_surplus_vars + num_artificial_vars)
        detail = constraint_types_detail[i]

        if detail['type'] == 'slack':
            full_row[s_ptr_offset + s_added_count] = 1.0
            basis_candidates.append(detail['name'])
            s_added_count += 1
        elif detail['type'] == 'surplus_artificial':
            full_row[e_ptr_offset + e_added_count] = -1.0 # Subtract surplus
            e_added_count += 1
            
            full_row[a_ptr_offset + a_added_count] = 1.0 # Add artificial
            basis_candidates.append(detail['artificial_name'])
            a_added_count += 1
        elif detail['type'] == 'equality_artificial':
            full_row[a_ptr_offset + a_added_count] = 1.0
            basis_candidates.append(detail['artificial_name'])
            a_added_count += 1
        A_matrix_rows.append(full_row)

    A_std = np.array(A_matrix_rows, dtype=float)
    b_std = np.array(temp_b_values_list, dtype=float)
    
    return A_std, b_std, var_names, basis_candidates, original_obj_coeffs_dict, \
           was_minimized, artificial_var_names_list


def create_hec_tableau(A_matrix, b_vector, var_names_all, 
                       basis_var_names_list, # List of names like ['e1', 'a2', 'a3']
                       obj_coeffs_map_for_Cj_row, # Dict: {'x1':10, 'e1':0, 'a2':-1 (for Max -W)}
                       obj_coeffs_map_for_Cb_col, # Dict: {'x1':10, 'e1':0, 'a2':-1 (for Max -W)}
                       is_phase_one=False): # is_phase_one is mostly for context, logic driven by maps
    
    num_constraints = A_matrix.shape[0]
    num_total_vars_in_A = A_matrix.shape[1]

    # Cj row values (for the very top row of the table)
    # Order must match var_names_all
    Cj_top_row_values = [obj_coeffs_map_for_Cj_row.get(var, 0.0) for var in var_names_all]

    # Cb column values (coefficients of basic variables in current objective)
    # Order must match basis_var_names_list
    Cb_column_values = [obj_coeffs_map_for_Cb_col.get(var_base, 0.0) for var_base in basis_var_names_list]

    # Calculate Zj row elements (for each variable column)
    Zj_coeffs_row = np.zeros(num_total_vars_in_A)
    for j_col_idx in range(num_total_vars_in_A): # For each variable x1, x2, e1...
        zj_val_for_col = 0
        for i_row_idx in range(num_constraints): # Sum over basic variables
            zj_val_for_col += Cb_column_values[i_row_idx] * A_matrix[i_row_idx, j_col_idx]
        Zj_coeffs_row[j_col_idx] = zj_val_for_col
    
    # Calculate Zj for RHS (current objective function value)
    Zj_rhs_value = np.dot(Cb_column_values, b_vector)

    # Calculate Cj - Zj row elements
    Cj_minus_Zj_row = np.array(Cj_top_row_values) - Zj_coeffs_row

    # --- Assemble DataFrame for internal use and easier data passing to formatter ---
    # Columns for the DataFrame (internal representation):
    # 'Base', 'Coef. Z', 'Var.base_Repeated', Var1, Var2, ..., 'bi'
    # 'Var.base_Repeated' is just to have a slot for "Zj" and "Cj-Zj" labels aligned like in HEC.

    df_data_list = []
    # Constraint rows
    for i in range(num_constraints):
        row_data = [basis_var_names_list[i]] + [Cb_column_values[i]] + [basis_var_names_list[i]] + \
                   list(A_matrix[i, :]) + [b_vector[i]]
        df_data_list.append(row_data)
    
    # Zj row
    df_data_list.append([""] + [np.nan] + ["Zj"] + list(Zj_coeffs_row) + [Zj_rhs_value])
    
    # Cj-Zj row
    df_data_list.append([""] + [np.nan] + ["Cj-Zj"] + list(Cj_minus_Zj_row) + [np.nan]) # bi for Cj-Zj is blank

    internal_df_columns = ["_Base_Actual", "_Coef.Z_Actual", "_Labels_Zj_CjZj"] + var_names_all + ["_bi_Actual"]
    tableau_df = pd.DataFrame(df_data_list, columns=internal_df_columns)

    # Store Cj_top_row_values as an attribute for the formatter
    tableau_df.attrs['Cj_top_row'] = Cj_top_row_values
    
    return tableau_df


def format_tableau_html_hec(tableau_df, iteration_str, description, 
                            pivot_r_idx_df=None, pivot_c_idx_df_internal=None, # pivot_c_idx_df_internal refers to index in var_names_all
                            entering_var_name=None, leaving_var_name=None):

    Cj_top_row = tableau_df.attrs.get('Cj_top_row', [])
    var_names_cols = [col for col in tableau_df.columns if col not in ["_Base_Actual", "_Coef.Z_Actual", "_Labels_Zj_CjZj", "_bi_Actual"]]
    
    # Determine the actual display column index for pivot_c if provided
    pivot_c_display_idx = None
    if pivot_c_idx_df_internal is not None:
        # pivot_c_idx_df_internal is the 0-based index within var_names_cols (x1, x2, e1...)
        # We need to map it to the display table column index:
        # Display cols: Base | Coef.Z | Var.base | x1 | x2 | ... | bi
        # So, add 3 to the internal var index.
        pivot_c_display_idx = pivot_c_idx_df_internal + 3


    html_parts = [f"<h3>{iteration_str}</h3>"]
    if entering_var_name and leaving_var_name:
        html_parts.append(f"<p><b>Entering Variable:</b> {entering_var_name}, <b>Leaving Variable:</b> {leaving_var_name}</p>")
    if description:
        html_parts.append(f"<p>{description.replace(chr(10), '<br>')}</p>")

    html_parts.append('<table class="dataframe table table-striped table-hover table-sm" border="1">')
    
    # 1. Cj Top Row (Coeff. dans Z)
    html_parts.append("<thead><tr>")
    html_parts.append("<th colspan='2'>Coeff. dans Z</th>") # Spans "Base" and "Coef.Z" columns visually
    html_parts.append("<th></th>") # Empty for "Var.base" column visual alignment
    for cj_val in Cj_top_row:
        html_parts.append(f"<th>{cj_val:.2f}</th>")
    html_parts.append("<th></th>") # Empty for "bi" column
    html_parts.append("</tr></thead>")

    # 2. Main Headers (Base, X1, X2 ..., bi) and (Coef.Z, Var.base)
    html_parts.append("<tbody>") # Start tbody for data rows
    html_parts.append("<tr>")
    html_parts.append("<th>Base</th>")
    html_parts.append("<th colspan='2'>Coef. Z     Var.base</th>") # Merged header concept
    for var_name in var_names_cols:
        html_parts.append(f"<th>{var_name}</th>")
    html_parts.append("<th>b<sub>i</sub></th>")
    html_parts.append("</tr>")

    # 3. Data Rows (Constraints, Zj, Cj-Zj)
    num_constraint_rows = len(tableau_df) - 2
    for r_idx, row_series in tableau_df.iterrows():
        html_parts.append("<tr>")
        
        # Cell 0: Base variable name (or blank for Zj, Cj-Zj)
        base_val_display = str(row_series["_Base_Actual"])
        html_parts.append(f"<td>{base_val_display}</td>")

        # Cell 1: Coef. Z (Cb) (or blank for Zj, Cj-Zj label column)
        coef_z_val = row_series["_Coef.Z_Actual"]
        coef_z_display = f"{coef_z_val:.2f}" if pd.notna(coef_z_val) else ""
        # Cell 2: Var.base repeated / Zj / Cj-Zj label
        label_val_display = str(row_series["_Labels_Zj_CjZj"])
        
        # Special handling for Zj and Cj-Zj rows to match HEC image layout for first three display columns
        if label_val_display == "Zj" or label_val_display == "Cj-Zj":
            html_parts.append("<td style='border-right:none;'></td>") # Blank under "Coef.Z" part
            html_parts.append(f"<td style='border-left:none; text-align:right; font-weight:bold;'>{label_val_display}</td>") # Label under "Var.base" part
        else: # Constraint row
            html_parts.append(f"<td style='border-right:none;'>{coef_z_display}</td>")
            html_parts.append(f"<td style='border-left:none;'>{label_val_display}</td>")


        # Variable columns and bi column
        for c_name_idx, var_col_name in enumerate(var_names_cols + ["_bi_Actual"]):
            cell_val = row_series[var_col_name]
            val_str = ""
            if pd.isna(cell_val): val_str = ""
            elif isinstance(cell_val, (float, np.floating)): val_str = f"{cell_val:.2f}"
            else: val_str = str(cell_val)

            style = ""
            current_display_col_idx = c_name_idx + 3 # +3 for Base, Coef.Z, Var.base display columns
            
            is_pivot_row = (pivot_r_idx_df is not None and r_idx == pivot_r_idx_df and r_idx < num_constraint_rows)
            is_pivot_col = (pivot_c_display_idx is not None and current_display_col_idx == pivot_c_display_idx and r_idx < num_constraint_rows)
            is_pivot_element = is_pivot_row and is_pivot_col

            if is_pivot_element:
                style = 'background-color: yellow; font-weight: bold; border: 1.5px solid red;'
            elif is_pivot_row and var_col_name != "_bi_Actual": # Highlight pivot row data cells
                style = 'background-color: lightyellow;'
            elif is_pivot_col: # Highlight pivot column data cells
                style = 'background-color: lightblue;'
            
            html_parts.append(f"<td style='{style}'>{val_str}</td>")
        html_parts.append("</tr>")

    html_parts.append("</tbody></table><hr>")
    return "".join(html_parts)


def find_pivot_column_hec(tableau_df, maximize_objective=True):
    # Cj-Zj values are in the last row, from the 4th column onwards (after _Base, _Coef.Z, _Labels), up to before _bi
    cj_minus_zj_values = tableau_df.iloc[-1, 3:-1].astype(float).values

    if maximize_objective:
        if np.all(cj_minus_zj_values <= EPSILON): return -1 # Optimal
        # This is index within cj_minus_zj_values (i.e., within var_names_all)
        pivot_col_idx_in_vars = np.argmax(cj_minus_zj_values) 
        if cj_minus_zj_values[pivot_col_idx_in_vars] <= EPSILON: return -1
    else: # Minimize (not primary path if always Max -Z)
        if np.all(cj_minus_zj_values >= -EPSILON): return -1
        pivot_col_idx_in_vars = np.argmin(cj_minus_zj_values)
        if cj_minus_zj_values[pivot_col_idx_in_vars] >= -EPSILON: return -1
            
    return pivot_col_idx_in_vars # This is the 0-based index for the variable in var_names_all list


def find_pivot_row_hec(tableau_df, pivot_col_var_idx): # pivot_col_var_idx is index in var_names_all
    # The actual column name for pivoting in the DataFrame
    var_names_cols = [col for col in tableau_df.columns if col not in ["_Base_Actual", "_Coef.Z_Actual", "_Labels_Zj_CjZj", "_bi_Actual"]]
    pivot_column_name_in_df = var_names_cols[pivot_col_var_idx]
    
    num_constraint_rows = len(tableau_df) - 2
    constraint_rows_data = tableau_df.iloc[:num_constraint_rows]
    
    pivot_column_coeffs = constraint_rows_data[pivot_column_name_in_df].astype(float).values
    bi_values = constraint_rows_data["_bi_Actual"].astype(float).values
    base_var_names_for_rows = constraint_rows_data["_Base_Actual"].values
    
    min_ratio = float('inf')
    pivot_row_actual_idx = -1 # This will be the direct row index in the full tableau_df
    valid_ratios_exist = False
    ratio_details_list = []

    for i in range(num_constraint_rows):
        coeff = pivot_column_coeffs[i]
        bi = bi_values[i]
        
        if coeff > EPSILON:
            valid_ratios_exist = True
            ratio = bi / coeff
            ratio_details_list.append(f"{base_var_names_for_rows[i]}: {bi:.2f}/{coeff:.2f} = {ratio:.2f}")
            if ratio < min_ratio - EPSILON:
                min_ratio = ratio
                pivot_row_actual_idx = i 
            elif abs(ratio - min_ratio) < EPSILON:
                if pivot_row_actual_idx == -1 or i < pivot_row_actual_idx :
                     pivot_row_actual_idx = i
        else:
            ratio_details_list.append(f"{base_var_names_for_rows[i]}: N/A (coeff ≤ 0)")

    if not valid_ratios_exist:
        return -1, "Unbounded: All pivot column elements in constraint rows are non-positive."
    if pivot_row_actual_idx == -1 :
        return -1, "Error: No valid pivot row found."

    ratio_desc = "Ratio Test (bi / Pivot Column Coeff):\n" + "\n".join(ratio_details_list) + \
                 f"\nMin Ratio = {min_ratio:.2f} (for row of {tableau_df.iloc[pivot_row_actual_idx]['_Base_Actual']})"
    return pivot_row_actual_idx, ratio_desc


def perform_pivot_operation_hec(current_A_matrix, current_b_vector, current_basis_names,
                                pivot_row_idx_in_A, pivot_col_var_idx_in_A): # These indices are for A_matrix

    new_A = current_A_matrix.copy()
    new_b = current_b_vector.copy()
    
    pivot_element = new_A[pivot_row_idx_in_A, pivot_col_var_idx_in_A]
    if abs(pivot_element) < EPSILON: raise ValueError("Pivot element is zero.")

    # Normalize pivot row
    new_A[pivot_row_idx_in_A, :] /= pivot_element
    new_b[pivot_row_idx_in_A] /= pivot_element

    # Update other constraint rows
    for r in range(new_A.shape[0]):
        if r != pivot_row_idx_in_A:
            factor = new_A[r, pivot_col_var_idx_in_A]
            new_A[r, :] -= factor * new_A[pivot_row_idx_in_A, :]
            new_b[r] -= factor * new_b[pivot_row_idx_in_A]
            
    # Update basis variable name list
    # Need var_names_all to get the entering variable's name from pivot_col_var_idx_in_A
    # This suggests var_names_all should be passed or accessible.
    # Let's assume entering_var_name is passed to this function or determined before.
    # For now, modify the main solver to determine entering_var_name from var_names_all & pivot_col_var_idx_in_A
    
    # This function will just return new_A, new_b. Basis update happens in main loop.
    return new_A, new_b


# ... (solve_simplex_problem_hec_style - needs to be adapted to use these revised functions)
def solve_simplex_problem_hec_style(obj_type_ui, obj_coeffs_str_ui, constraints_inputs_ui, 
                                    num_vars_ui, num_constraints_ui):
    steps_html = ""
    MAX_ITERATIONS = 25
    iteration_ph1 = 0 
    current_W_val = 0.0  # Initialize current_W_val to a default value

    try:
        obj_coeffs_list_ui = [float(x.strip()) for x in obj_coeffs_str_ui.split(',')]
        num_vars_val = int(num_vars_ui)
        num_constraints_val = int(num_constraints_ui)
        # ... (rest of input parsing as before) ...
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
            parsed_constraints_list.append({'coeffs': coeffs, 'op': op, 'rhs': float(rhs_str)})

    except Exception as e:
        return f"<p style='color:red'>Input Parsing Error: {e}</p>", None

    A_initial, b_initial, var_names_all, current_basis_names, orig_obj_coeffs_map, \
    was_minimized, artificial_var_names = \
        standardize_problem(obj_type_ui, obj_coeffs_list_ui, parsed_constraints_list, num_vars_val)

    steps_html += f"<h2>Initial Setup</h2><p>Variables: {', '.join(var_names_all)}</p>"
    steps_html += f"<p>Artificial Vars: {', '.join(artificial_var_names) if artificial_var_names else 'None'}</p><hr>"

    current_A_matrix = A_initial.copy()
    current_b_vector = b_initial.copy()
    # current_basis_names is already set from standardize_problem

    # --- Phase I ---
    if artificial_var_names:
        steps_html += "<h2>Phase I: Minimize Sum of Artificial Variables (by Max -W)</h2>"
        
        # Cj for Phase I objective (Max -W): -1 for artificial vars, 0 for others
        # This is for the Cj_top_row
        cj_map_phase_I = {var: (-1.0 if var in artificial_var_names else 0.0) for var in var_names_all}
        # Cb for Phase I objective (coefficients of basic vars in -W objective)
        # This is also -1 if basic var is artificial, 0 otherwise.
        cb_map_phase_I = cj_map_phase_I # In Phase I, Cj and Cb for a given var type are the same.
        
        iteration_ph1 = 0
        while iteration_ph1 <= MAX_ITERATIONS: # Use <= to allow MAX_ITERATIONS attempts
            tableau = create_hec_tableau(current_A_matrix, current_b_vector, var_names_all, 
                                         current_basis_names, cj_map_phase_I, cb_map_phase_I, is_phase_one=True)
            
            # W value: Zj value in '_bi_Actual' column for Zj row. This is -W. So W = -value.
            current_W_val = -float(tableau[tableau["_Labels_Zj_CjZj"] == "Zj"].iloc[0]["_bi_Actual"])
            
            pivot_col_var_idx = find_pivot_column_hec(tableau, maximize_objective=True)

            if pivot_col_var_idx == -1: # Optimal for Phase I
                if abs(current_W_val) > EPSILON:
                    steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1} (Final)", 
                                                          f"Phase I Optimal. W = {current_W_val:.4f}. Problem Infeasible.")
                    return steps_html, {"status": "Infeasible", "message": f"Artificial variables sum to {current_W_val:.4f}."}
                else:
                    steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1} (Final)", 
                                                          f"Phase I Optimal. W = {current_W_val:.4f}. Feasible. Proceed to Phase II.")
                    break 
            
            pivot_row_idx_in_A, ratio_desc = find_pivot_row_hec(tableau, pivot_col_var_idx)

            if pivot_row_idx_in_A == -1:
                steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1}", 
                                                      f"{ratio_desc}\nPhase I Unbounded (Error). W = {current_W_val:.4f}")
                return steps_html, {"status": "Error", "message": "Phase I Unbounded."}

            entering_var_name = var_names_all[pivot_col_var_idx]
            leaving_var_name = current_basis_names[pivot_row_idx_in_A]
            
            steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1}", ratio_desc,
                                                  pivot_r_idx_df=pivot_row_idx_in_A, 
                                                  pivot_c_idx_df_internal=pivot_col_var_idx,
                                                  entering_var_name=entering_var_name, leaving_var_name=leaving_var_name)

            current_A_matrix, current_b_vector = \
                perform_pivot_operation_hec(current_A_matrix, current_b_vector, current_basis_names,
                                            pivot_row_idx_in_A, pivot_col_var_idx)
            current_basis_names[pivot_row_idx_in_A] = entering_var_name # Update basis
            
            iteration_ph1 += 1
            if iteration_ph1 > MAX_ITERATIONS: # Check after increment
                return steps_html + "<p style='color:red'>Phase I: Max iterations reached.</p>", {"status": "Max Iterations (Phase I)"}
        
        if iteration_ph1 > MAX_ITERATIONS and abs(current_W_val) > EPSILON :
             return steps_html + "<p style='color:red'>Phase I: Max iterations reached and W > 0.</p>", {"status": "Max Iterations (Phase I), W > 0"}

    # --- Phase II ---
    steps_html += "<h2>Phase II: Solve Original Problem</h2>"
    
    cj_map_phase_II = {}
    for var in var_names_all:
        original_coeff = orig_obj_coeffs_map.get(var, 0.0) # Try direct match first (e.g. x1)
        if var not in orig_obj_coeffs_map and var.startswith('e'): # Slack/surplus vars have 0 coeff
            original_coeff = 0.0
        elif var in artificial_var_names: # Artificial vars have 0 coeff in Phase II obj
            original_coeff = 0.0
        
        cj_map_phase_II[var] = -original_coeff if was_minimized else original_coeff
            
    iteration_ph2 = 0
    while iteration_ph2 <= MAX_ITERATIONS:
        cb_map_phase_II = {var_b: cj_map_phase_II.get(var_b, 0.0) for var_b in current_basis_names}

        tableau = create_hec_tableau(current_A_matrix, current_b_vector, var_names_all,
                                     current_basis_names, cj_map_phase_II, cb_map_phase_II, is_phase_one=False)

        pivot_col_var_idx = find_pivot_column_hec(tableau, maximize_objective=True)

        if pivot_col_var_idx == -1: # Optimal for Phase II
            # Zj_rhs_value is for the objective being maximized (original Z or -Z_min)
            final_obj_val_maximized = float(tableau[tableau["_Labels_Zj_CjZj"] == "Zj"].iloc[0]["_bi_Actual"])
            actual_obj_val = -final_obj_val_maximized if was_minimized else final_obj_val_maximized
            
            solution_vars = {}
            num_decision_vars = len(orig_obj_coeffs_map) # Count of x1, x2...
            for i in range(num_decision_vars): 
                var_name_orig_decision = f"x{i+1}"
                if var_name_orig_decision in current_basis_names:
                    row_idx_in_basis = current_basis_names.index(var_name_orig_decision)
                    solution_vars[var_name_orig_decision] = float(current_b_vector[row_idx_in_basis])
                else:
                    solution_vars[var_name_orig_decision] = 0.0
            
            steps_html += format_tableau_html_hec(tableau, f"Phase II - Iteration {iteration_ph2} (Optimal)", 
                                                  f"Optimal Solution Found. Z = {actual_obj_val:.4f}")
            return steps_html, {"status": "Optimal", "value": actual_obj_val, "variables": solution_vars}

        pivot_row_idx_in_A, ratio_desc = find_pivot_row_hec(tableau, pivot_col_var_idx)

        if pivot_row_idx_in_A == -1:
            steps_html += format_tableau_html_hec(tableau, f"Phase II - Iteration {iteration_ph2}", 
                                                  f"{ratio_desc}\nProblem Unbounded.")
            return steps_html, {"status": "Unbounded", "message": ratio_desc}

        entering_var_name = var_names_all[pivot_col_var_idx]
        leaving_var_name = current_basis_names[pivot_row_idx_in_A]
        
        steps_html += format_tableau_html_hec(tableau, f"Phase II - Iteration {iteration_ph2}", ratio_desc,
                                              pivot_r_idx_df=pivot_row_idx_in_A, 
                                              pivot_c_idx_df_internal=pivot_col_var_idx,
                                              entering_var_name=entering_var_name, leaving_var_name=leaving_var_name)

        current_A_matrix, current_b_vector = \
            perform_pivot_operation_hec(current_A_matrix, current_b_vector, current_basis_names,
                                        pivot_row_idx_in_A, pivot_col_var_idx)
        current_basis_names[pivot_row_idx_in_A] = entering_var_name # Update basis
            
        iteration_ph2 += 1
        if iteration_ph2 > MAX_ITERATIONS:
            return steps_html + "<p style='color:red'>Phase II: Max iterations reached.</p>", {"status": "Max Iterations (Phase II)"}
    
    return steps_html + "<p style='color:red'>Solver finished without a definitive result.</p>", {"status": "Unknown Error"}