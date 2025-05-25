# simplex_logic.py
import numpy as np
import pandas as pd

EPSILON = 1e-9

# --- standardize_problem, create_hec_tableau, format_tableau_html_hec ---
# --- find_pivot_column_hec (Primal), find_pivot_row_hec (Primal) ---
# --- perform_pivot_operation_hec ---
# (These functions are assumed to be the same as the last correct version you have,
#  I will re-paste them for completeness but focus on the new dual functions)

def standardize_problem(obj_type, obj_coeffs, constraints_data, num_vars, obj_type_ui):
    # (Same as before - from your previous message with the full file content)
    obj_coeffs = [float(c) for c in obj_coeffs]
    original_obj_coeffs_dict = {f"x{i+1}": obj_coeffs[i] for i in range(num_vars)}
    was_minimized = (obj_type == "Minimize")
    var_names = [f"x{i+1}" for i in range(num_vars)]
    num_decision_vars = num_vars
    s_idx_counter = 0
    e_idx_counter = 0 
    a_idx_counter = 0
    basis_candidates = [] 
    artificial_var_names_list = []
    temp_A_coeffs_list = []
    temp_b_values_list = []
    temp_ops_list = []
    constraint_types_detail = []

    for constr in constraints_data:
        coeffs = [float(c) for c in constr['coeffs']]
        rhs = float(constr['rhs'])
        op = constr['op']
        # DO NOT flip RHS sign for Dual Simplex here; Dual Simplex *starts* with negative RHS.
        # However, for Primal Simplex standardization, this was done.
        # For a unified approach, maybe this flipping should be conditional or handled differently.
        # For now, let's assume standard primal setup, and Dual Simplex will find negative b_i if they exist.
        if obj_type_ui != "Dual Simplex Initial": # Only flip for primal setup
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
            e_idx_counter += 1
            a_idx_counter += 1
            constraint_types_detail.append({'type': 'surplus_artificial', 
                                            'surplus_name': f"e{s_idx_counter + e_idx_counter}",
                                            'artificial_name': f"a{a_idx_counter}"})
        elif op == "=":
            a_idx_counter += 1
            constraint_types_detail.append({'type': 'equality_artificial', 
                                            'artificial_name': f"a{a_idx_counter}"})
    
    num_slack_vars = s_idx_counter
    num_surplus_vars = e_idx_counter
    num_artificial_vars = a_idx_counter
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

    A_matrix_rows = []
    s_ptr_offset = num_decision_vars
    e_ptr_offset = num_decision_vars + num_slack_vars 
    a_ptr_offset = num_decision_vars + num_slack_vars + num_surplus_vars
    s_added_count = 0
    e_added_count = 0
    a_added_count = 0

    # Reset naming index for correct association during basis candidate selection
    current_e_naming_idx_for_basis = 0 
    current_a_naming_idx_for_basis = 0

    for i in range(len(temp_A_coeffs_list)):
        row_coeffs = temp_A_coeffs_list[i]
        full_row = list(row_coeffs) + [0.0] * (num_slack_vars + num_surplus_vars + num_artificial_vars)
        detail = constraint_types_detail[i]

        if detail['type'] == 'slack':
            current_e_naming_idx_for_basis += 1
            slack_var_name = f"e{current_e_naming_idx_for_basis}"
            full_row[s_ptr_offset + s_added_count] = 1.0
            basis_candidates.append(slack_var_name)
            s_added_count += 1
        elif detail['type'] == 'surplus_artificial':
            current_e_naming_idx_for_basis += 1 # for surplus 'e'
            surplus_var_name = f"e{current_e_naming_idx_for_basis}"
            full_row[e_ptr_offset + e_added_count] = -1.0
            e_added_count += 1
            
            current_a_naming_idx_for_basis +=1 # for artificial 'a'
            artif_var_name = f"a{current_a_naming_idx_for_basis}"
            full_row[a_ptr_offset + a_added_count] = 1.0
            basis_candidates.append(artif_var_name)
            a_added_count += 1
        elif detail['type'] == 'equality_artificial':
            current_a_naming_idx_for_basis +=1
            artif_var_name = f"a{current_a_naming_idx_for_basis}"
            full_row[a_ptr_offset + a_added_count] = 1.0
            basis_candidates.append(artif_var_name)
            a_added_count += 1
        A_matrix_rows.append(full_row)

    A_std = np.array(A_matrix_rows, dtype=float)
    b_std = np.array(temp_b_values_list, dtype=float)
    
    return A_std, b_std, var_names, basis_candidates, original_obj_coeffs_dict, \
           was_minimized, artificial_var_names_list, obj_type # Pass obj_type for conditional logic


def create_hec_tableau(A_matrix, b_vector, var_names_all, 
                       basis_var_names_list, 
                       obj_coeffs_map_for_Cj_row, 
                       obj_coeffs_map_for_Cb_col, 
                       is_phase_one=False):
    # (Same as before)
    num_constraints = A_matrix.shape[0]
    num_total_vars_in_A = A_matrix.shape[1]
    Cj_top_row_values = [obj_coeffs_map_for_Cj_row.get(var, 0.0) for var in var_names_all]
    Cb_column_values = [obj_coeffs_map_for_Cb_col.get(var_base, 0.0) for var_base in basis_var_names_list]
    Zj_coeffs_row = np.zeros(num_total_vars_in_A)
    for j_col_idx in range(num_total_vars_in_A):
        zj_val_for_col = 0
        for i_row_idx in range(num_constraints):
            zj_val_for_col += Cb_column_values[i_row_idx] * A_matrix[i_row_idx, j_col_idx]
        Zj_coeffs_row[j_col_idx] = zj_val_for_col
    Zj_rhs_value = np.dot(Cb_column_values, b_vector)
    Cj_minus_Zj_row = np.array(Cj_top_row_values) - Zj_coeffs_row
    df_data_list = []
    for i in range(num_constraints):
        row_data = [basis_var_names_list[i]] + [Cb_column_values[i]] + [basis_var_names_list[i]] + \
                   list(A_matrix[i, :]) + [b_vector[i]]
        df_data_list.append(row_data)
    df_data_list.append([""] + [np.nan] + ["Zj"] + list(Zj_coeffs_row) + [Zj_rhs_value])
    df_data_list.append([""] + [np.nan] + ["Cj-Zj"] + list(Cj_minus_Zj_row) + [np.nan])
    internal_df_columns = ["_Base_Actual", "_Coef.Z_Actual", "_Labels_Zj_CjZj"] + var_names_all + ["_bi_Actual"]
    tableau_df = pd.DataFrame(df_data_list, columns=internal_df_columns)
    tableau_df.attrs['Cj_top_row'] = Cj_top_row_values
    return tableau_df

def format_tableau_html_hec(tableau_df, iteration_str, description, 
                            pivot_r_idx_df=None, pivot_c_idx_df_internal=None,
                            entering_var_name=None, leaving_var_name=None):
    # (Same as before)
    Cj_top_row = tableau_df.attrs.get('Cj_top_row', [])
    var_names_cols = [col for col in tableau_df.columns if col not in ["_Base_Actual", "_Coef.Z_Actual", "_Labels_Zj_CjZj", "_bi_Actual"]]
    pivot_c_display_idx = None
    if pivot_c_idx_df_internal is not None:
        pivot_c_display_idx = pivot_c_idx_df_internal + 3
    html_parts = [f"<h3>{iteration_str}</h3>"]
    if entering_var_name and leaving_var_name:
        html_parts.append(f"<p><b>Entering Variable:</b> {entering_var_name}, <b>Leaving Variable:</b> {leaving_var_name}</p>")
    if description:
        html_parts.append(f"<p>{description.replace(chr(10), '<br>')}</p>")
    html_parts.append('<table class="dataframe table table-striped table-hover table-sm" border="1">')
    html_parts.append("<thead><tr>")
    html_parts.append("<th colspan='2'>Coeff. dans Z</th>") 
    html_parts.append("<th></th>") 
    for cj_val in Cj_top_row:
        html_parts.append(f"<th>{cj_val:.2f}</th>")
    html_parts.append("<th></th>")
    html_parts.append("</tr></thead>")
    html_parts.append("<tbody>")
    html_parts.append("<tr>")
    html_parts.append("<th>Base</th>")
    html_parts.append("<th colspan='2'>Coef. Z     Var.base</th>")
    for var_name in var_names_cols:
        html_parts.append(f"<th>{var_name}</th>")
    html_parts.append("<th>b<sub>i</sub></th>")
    html_parts.append("</tr>")
    num_constraint_rows = len(tableau_df) - 2
    for r_idx, row_series in tableau_df.iterrows():
        html_parts.append("<tr>")
        base_val_display = str(row_series["_Base_Actual"])
        html_parts.append(f"<td>{base_val_display}</td>")
        coef_z_val = row_series["_Coef.Z_Actual"]
        coef_z_display = f"{coef_z_val:.2f}" if pd.notna(coef_z_val) else ""
        label_val_display = str(row_series["_Labels_Zj_CjZj"])
        if label_val_display == "Zj" or label_val_display == "Cj-Zj":
            html_parts.append("<td style='border-right:none;'></td>")
            html_parts.append(f"<td style='border-left:none; text-align:right; font-weight:bold;'>{label_val_display}</td>")
        else:
            html_parts.append(f"<td style='border-right:none;'>{coef_z_display}</td>")
            html_parts.append(f"<td style='border-left:none;'>{label_val_display}</td>")
        for c_name_idx, var_col_name in enumerate(var_names_cols + ["_bi_Actual"]):
            cell_val = row_series[var_col_name]
            val_str = ""
            if pd.isna(cell_val): val_str = ""
            elif isinstance(cell_val, (float, np.floating)): val_str = f"{cell_val:.2f}"
            else: val_str = str(cell_val)
            style = ""
            current_display_col_idx = c_name_idx + 3
            is_pivot_row = (pivot_r_idx_df is not None and r_idx == pivot_r_idx_df and r_idx < num_constraint_rows)
            is_pivot_col = (pivot_c_display_idx is not None and current_display_col_idx == pivot_c_display_idx and r_idx < num_constraint_rows)
            is_pivot_element = is_pivot_row and is_pivot_col
            if is_pivot_element: style = 'background-color: yellow; font-weight: bold; border: 1.5px solid red;'
            elif is_pivot_row and var_col_name != "_bi_Actual": style = 'background-color: lightyellow;'
            elif is_pivot_col: style = 'background-color: lightblue;'
            html_parts.append(f"<td style='{style}'>{val_str}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table><hr>")
    return "".join(html_parts)

def find_pivot_column_hec(tableau_df, maximize_objective=True): # For Primal Simplex
    # (Same as before)
    cj_minus_zj_values = tableau_df.iloc[-1, 3:-1].astype(float).values
    if maximize_objective:
        if np.all(cj_minus_zj_values <= EPSILON): return -1
        pivot_col_idx_in_vars = np.argmax(cj_minus_zj_values) 
        if cj_minus_zj_values[pivot_col_idx_in_vars] <= EPSILON: return -1
    else: 
        if np.all(cj_minus_zj_values >= -EPSILON): return -1
        pivot_col_idx_in_vars = np.argmin(cj_minus_zj_values)
        if cj_minus_zj_values[pivot_col_idx_in_vars] >= -EPSILON: return -1
    return pivot_col_idx_in_vars

def find_pivot_row_hec(tableau_df, pivot_col_var_idx): # For Primal Simplex
    # (Same as before)
    var_names_cols = [col for col in tableau_df.columns if col not in ["_Base_Actual", "_Coef.Z_Actual", "_Labels_Zj_CjZj", "_bi_Actual"]]
    pivot_column_name_in_df = var_names_cols[pivot_col_var_idx]
    num_constraint_rows = len(tableau_df) - 2
    constraint_rows_data = tableau_df.iloc[:num_constraint_rows]
    pivot_column_coeffs = constraint_rows_data[pivot_column_name_in_df].astype(float).values
    bi_values = constraint_rows_data["_bi_Actual"].astype(float).values
    base_var_names_for_rows = constraint_rows_data["_Base_Actual"].values
    min_ratio = float('inf')
    pivot_row_actual_idx = -1
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
    if not valid_ratios_exist: return -1, "Unbounded (Primal): All pivot column elements non-positive."
    if pivot_row_actual_idx == -1 : return -1, "Error (Primal): No valid pivot row."
    ratio_desc = "Ratio Test (Primal) (bi / Pivot Col Coeff):\n" + "\n".join(ratio_details_list) + \
                 f"\nMin Ratio = {min_ratio:.2f} (for row of {tableau_df.iloc[pivot_row_actual_idx]['_Base_Actual']})"
    return pivot_row_actual_idx, ratio_desc

def perform_pivot_operation_hec(current_A_matrix, current_b_vector, 
                                pivot_row_idx_in_A, pivot_col_var_idx_in_A):
    # (Same as before - Note: current_basis_names was removed as it's updated in the main loop)
    new_A = current_A_matrix.copy()
    new_b = current_b_vector.copy()
    pivot_element = new_A[pivot_row_idx_in_A, pivot_col_var_idx_in_A]
    if abs(pivot_element) < EPSILON: raise ValueError("Pivot element is zero.")
    new_A[pivot_row_idx_in_A, :] /= pivot_element
    new_b[pivot_row_idx_in_A] /= pivot_element
    for r in range(new_A.shape[0]):
        if r != pivot_row_idx_in_A:
            factor = new_A[r, pivot_col_var_idx_in_A]
            new_A[r, :] -= factor * new_A[pivot_row_idx_in_A, :]
            new_b[r] -= factor * new_b[pivot_row_idx_in_A]
    return new_A, new_b

# --- NEW DUAL SIMPLEX PIVOT FUNCTIONS ---
def find_dual_pivot_row_hec(tableau_df):
    """ Chooses leaving variable for Dual Simplex. Most negative bi. """
    bi_values = tableau_df.iloc[:-2]["_bi_Actual"].astype(float).values # RHS of constraint rows
    
    if np.all(bi_values >= -EPSILON): # Check against -EPSILON for near-zero positive values
        return -1, "Primal feasible. Optimal solution found for Dual Simplex." # All RHS non-negative

    # Find index of the most negative bi value
    # np.argmin will find the first occurrence if multiple identical minimums
    pivot_row_df_idx = np.argmin(bi_values) 
    
    if bi_values[pivot_row_df_idx] >= -EPSILON: # Should be caught by np.all check, but as safeguard
        return -1, "Primal feasible (safeguard). Optimal solution found."

    leaving_var_name = tableau_df.iloc[pivot_row_df_idx]["_Base_Actual"]
    description = f"Pivot Row Selection (Dual): Smallest RHS is {bi_values[pivot_row_df_idx]:.2f} for basic variable {leaving_var_name}."
    return pivot_row_df_idx, description # This is the direct index in the full tableau_df

def find_dual_pivot_column_hec(tableau_df, pivot_row_df_idx):
    """ Chooses entering variable for Dual Simplex. Min ratio |(Cj-Zj)/a_rj| for a_rj < 0. """
    pivot_row_coeffs = tableau_df.iloc[pivot_row_df_idx, 3:-1].astype(float).values # Coeffs in pivot row (for x, e, a vars)
    cj_minus_zj_values = tableau_df.iloc[-1, 3:-1].astype(float).values # Cj-Zj row (for x, e, a vars)

    min_abs_ratio = float('inf')
    pivot_col_var_idx = -1 # This will be index within var_names_all
    valid_candidates_exist = False
    ratio_details_list = []
    var_names_cols = [col for col in tableau_df.columns if col not in ["_Base_Actual", "_Coef.Z_Actual", "_Labels_Zj_CjZj", "_bi_Actual"]]


    for j, a_rj in enumerate(pivot_row_coeffs):
        var_name = var_names_cols[j]
        if a_rj < -EPSILON: # Denominator must be strictly negative
            valid_candidates_exist = True
            cj_zj_j = cj_minus_zj_values[j]
            # Optimality condition (Cj-Zj <= 0 for max) should already hold.
            # If Cj-Zj is positive, this rule might select it if a_rj is negative,
            # potentially breaking dual feasibility. This implies the tableau wasn't dual feasible to start.
            # Standard rule expects Cj-Zj_j <= 0.
            if cj_zj_j > EPSILON : # Warn if trying to pivot on a Cj-Zj that's not maintaining dual feasibility
                ratio_details_list.append(f"{var_name}: Cj-Zj > 0 ({cj_zj_j:.2f}), ratio skipped for dual feasibility maintenance.")
                continue

            ratio = cj_zj_j / a_rj # This ratio will be non-negative if Cj-Zj_j <=0 and a_rj < 0
            ratio_details_list.append(f"{var_name}: ({cj_zj_j:.2f})/({a_rj:.2f}) = {ratio:.2f}")

            if ratio < min_abs_ratio - EPSILON : # Smallest non-negative ratio (as per |(Cj-Zj)/a_rj|)
                                                 # Since (Cj-Zj) likely <=0 and a_rj < 0, ratio >=0. Smallest positive.
                min_abs_ratio = ratio
                pivot_col_var_idx = j
            # Simple tie-breaking: smallest index of variable
            elif abs(ratio - min_abs_ratio) < EPSILON:
                if pivot_col_var_idx == -1 or j < pivot_col_var_idx:
                    pivot_col_var_idx = j
        else:
            ratio_details_list.append(f"{var_name}: N/A (coeff in pivot row a_rj ≥ 0)")


    if not valid_candidates_exist:
        return -1, "Primal Infeasible (Dual Unbounded): All coefficients in pivot row ≥ 0."
    
    if pivot_col_var_idx == -1: # Should be caught by valid_candidates_exist
         return -1, "Error (Dual): No valid pivot column found."

    description = "Pivot Column Selection (Dual) - Ratios (Cj-Zj)/a_rj for a_rj < 0:\n" + \
                  "\n".join(ratio_details_list) + \
                  f"\nSelected: Min Positive Ratio = {min_abs_ratio:.2f} for variable {var_names_cols[pivot_col_var_idx]}"
    return pivot_col_var_idx, description


# --- Main Solver Orchestrator ---
def solve_simplex_main(obj_type_ui, obj_coeffs_str_ui, constraints_inputs_ui, 
                       num_vars_ui, num_constraints_ui, method_type):
    
    steps_html = ""
    MAX_ITERATIONS = 25 
    
    # --- 1. Input Parsing & Standardization ---
    try:
        obj_coeffs_list_ui = [float(x.strip()) for x in obj_coeffs_str_ui.split(',')]
        num_vars_val = int(num_vars_ui)
        num_constraints_val = int(num_constraints_ui)
        if len(obj_coeffs_list_ui) != num_vars_val:
            return f"<p style='color:red'>Input Error: Mismatch in objective coefficients and number of variables.</p>", None
        parsed_constraints_list = []
        for i in range(num_constraints_val):
            # ... (constraint parsing as before)
            coeffs_str = constraints_inputs_ui[f"const_{i}_coeffs"]
            op = constraints_inputs_ui[f"const_{i}_op"]
            rhs_str = constraints_inputs_ui[f"const_{i}_rhs"]
            coeffs = [float(x.strip()) for x in coeffs_str.split(',')]
            if len(coeffs) != num_vars_val:
                 return f"<p style='color:red'>Input Error: Constraint {i+1} coefficient mismatch.</p>", None
            parsed_constraints_list.append({'coeffs': coeffs, 'op': op, 'rhs': float(rhs_str)})
    except Exception as e:
        return f"<p style='color:red'>Input Parsing Error: {e}</p>", None

    # Modify standardize_problem to accept obj_type_ui for conditional RHS flipping
    global obj_type_ui_global # Temporary hack for standardize_problem access
    A_initial, b_initial, var_names_all, current_basis_names, orig_obj_coeffs_map, \
    was_minimized, artificial_var_names, _ = \
        standardize_problem(obj_type_ui, obj_coeffs_list_ui, parsed_constraints_list, num_vars_val, 
                            obj_type_ui if method_type != "Dual Simplex" else "Dual Simplex Initial")

    steps_html += f"<h2>Initial Problem Setup</h2><p>Selected Method: {method_type}</p>"
    steps_html += f"<p>Variables: {', '.join(var_names_all)}</p>"
    steps_html += f"<p>Initial Basis Candidates from Standardization: {', '.join(current_basis_names)}</p>"
    steps_html += f"<p>Artificial Vars: {', '.join(artificial_var_names) if artificial_var_names else 'None'}</p><hr>"

    current_A_matrix = A_initial.copy()
    current_b_vector = b_initial.copy()

    # --- 2. Primal Simplex Path (with Two-Phase) ---
    if method_type == "Primal Simplex (Two-Phase)":
        # --- Phase I (if needed for Primal Simplex) ---
        if artificial_var_names:
            # Initialize current_W_val to ensure it is always defined
            current_W_val = 0.0
            # ... (Phase I logic from solve_simplex_problem_hec_style, using Primal pivot rules)
            steps_html += "<h2>Phase I: Minimize Sum of Artificial Variables (by Max -W)</h2>"
            cj_map_phase_I = {var: (-1.0 if var in artificial_var_names else 0.0) for var in var_names_all}
            cb_map_phase_I = cj_map_phase_I
            iteration_ph1 = 0
            # (Loop for Phase I using find_pivot_column_hec and find_pivot_row_hec)
            while iteration_ph1 <= MAX_ITERATIONS:
                tableau = create_hec_tableau(current_A_matrix, current_b_vector, var_names_all, 
                                             current_basis_names, cj_map_phase_I, cb_map_phase_I, is_phase_one=True)
                current_W_val = -float(tableau[tableau["_Labels_Zj_CjZj"] == "Zj"].iloc[0]["_bi_Actual"])
                pivot_col_var_idx = find_pivot_column_hec(tableau, maximize_objective=True) # Primal rule

                if pivot_col_var_idx == -1: 
                    # ... (optimality/infeasibility check for Phase I)
                    if abs(current_W_val) > EPSILON:
                        steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1} (Final)", f"Phase I Optimal. W = {current_W_val:.4f}. Problem Infeasible.")
                        return steps_html, {"status": "Infeasible", "message": f"Artificial vars sum to {current_W_val:.4f}."}
                    else:
                        steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1} (Final)", f"Phase I Optimal. W = {current_W_val:.4f}. Feasible. Proceed to Phase II.")
                        break 
                
                pivot_row_idx_in_A, ratio_desc = find_pivot_row_hec(tableau, pivot_col_var_idx) # Primal rule
                if pivot_row_idx_in_A == -1:
                    # ... (unbounded check for Phase I)
                    steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1}", f"{ratio_desc}\nPhase I Unbounded (Error). W = {current_W_val:.4f}")
                    return steps_html, {"status": "Error", "message": "Phase I Unbounded for Primal."}

                entering_var_name = var_names_all[pivot_col_var_idx]
                leaving_var_name = current_basis_names[pivot_row_idx_in_A]
                steps_html += format_tableau_html_hec(tableau, f"Phase I - Iteration {iteration_ph1}", ratio_desc,
                                                      pivot_r_idx_df=pivot_row_idx_in_A, pivot_c_idx_df_internal=pivot_col_var_idx,
                                                      entering_var_name=entering_var_name, leaving_var_name=leaving_var_name)
                current_A_matrix, current_b_vector = perform_pivot_operation_hec(current_A_matrix, current_b_vector, pivot_row_idx_in_A, pivot_col_var_idx)
                current_basis_names[pivot_row_idx_in_A] = entering_var_name
                iteration_ph1 += 1
                if iteration_ph1 > MAX_ITERATIONS: return steps_html + "<p style='color:red'>Phase I: Max iterations.</p>", {"status": "Max Iterations (Phase I Primal)"}
            if iteration_ph1 > MAX_ITERATIONS and abs(current_W_val) > EPSILON: return steps_html + "<p style='color:red'>Phase I: Max iter & W>0.</p>", {"status": "Max Iter (Ph I Primal), W>0"}
        
        # --- Phase II (for Primal Simplex) ---
        steps_html += "<h2>Phase II: Solve Original Problem (Primal Simplex)</h2>"
        # (Setup Cj, Cb for Phase II and loop using Primal pivot rules)
        cj_map_phase_II = {} # As defined in previous solve_simplex_problem_hec_style
        for var in var_names_all:
            original_coeff = orig_obj_coeffs_map.get(var, 0.0) 
            if var not in orig_obj_coeffs_map and var.startswith('e'): original_coeff = 0.0
            elif var in artificial_var_names: original_coeff = 0.0
            cj_map_phase_II[var] = -original_coeff if was_minimized else original_coeff
        
        iteration_ph2 = 0
        while iteration_ph2 <= MAX_ITERATIONS:
            cb_map_phase_II = {var_b: cj_map_phase_II.get(var_b, 0.0) for var_b in current_basis_names}
            tableau = create_hec_tableau(current_A_matrix, current_b_vector, var_names_all,
                                         current_basis_names, cj_map_phase_II, cb_map_phase_II)
            pivot_col_var_idx = find_pivot_column_hec(tableau, maximize_objective=True) # Primal rule

            if pivot_col_var_idx == -1: 
                # ... (Optimal solution found for Primal Phase II)
                final_obj_val_maximized = float(tableau[tableau["_Labels_Zj_CjZj"] == "Zj"].iloc[0]["_bi_Actual"])
                actual_obj_val = -final_obj_val_maximized if was_minimized else final_obj_val_maximized
                solution_vars = {} # Populate as before
                num_decision_vars = len(orig_obj_coeffs_map)
                for i in range(num_decision_vars): 
                    var_name_orig_decision = f"x{i+1}"
                    if var_name_orig_decision in current_basis_names:
                        row_idx_in_basis = current_basis_names.index(var_name_orig_decision)
                        solution_vars[var_name_orig_decision] = float(current_b_vector[row_idx_in_basis])
                    else: solution_vars[var_name_orig_decision] = 0.0
                steps_html += format_tableau_html_hec(tableau, f"Phase II - Iteration {iteration_ph2} (Optimal)", f"Optimal Solution. Z = {actual_obj_val:.4f}")
                return steps_html, {"status": "Optimal", "value": actual_obj_val, "variables": solution_vars}

            pivot_row_idx_in_A, ratio_desc = find_pivot_row_hec(tableau, pivot_col_var_idx) # Primal rule
            if pivot_row_idx_in_A == -1:
                # ... (Unbounded for Primal Phase II)
                steps_html += format_tableau_html_hec(tableau, f"Phase II - Iteration {iteration_ph2}", f"{ratio_desc}\nProblem Unbounded (Primal).")
                return steps_html, {"status": "Unbounded", "message": ratio_desc}
            
            entering_var_name = var_names_all[pivot_col_var_idx]
            leaving_var_name = current_basis_names[pivot_row_idx_in_A]
            steps_html += format_tableau_html_hec(tableau, f"Phase II - Iteration {iteration_ph2}", ratio_desc,
                                                  pivot_r_idx_df=pivot_row_idx_in_A, pivot_c_idx_df_internal=pivot_col_var_idx,
                                                  entering_var_name=entering_var_name, leaving_var_name=leaving_var_name)
            current_A_matrix, current_b_vector = perform_pivot_operation_hec(current_A_matrix, current_b_vector, pivot_row_idx_in_A, pivot_col_var_idx)
            current_basis_names[pivot_row_idx_in_A] = entering_var_name
            iteration_ph2 += 1
            if iteration_ph2 > MAX_ITERATIONS: return steps_html + "<p style='color:red'>Phase II: Max iter.</p>", {"status": "Max Iter (Ph II Primal)"}
        # Fallback for Primal path if loop finishes unexpectedly
        return steps_html + "<p style='color:red'>Primal Simplex did not conclude.</p>", {"status": "Error (Primal Simplex)"}


    # --- 3. Dual Simplex Path ---
    elif method_type == "Dual Simplex":
        steps_html += "<h2>Dual Simplex Method</h2>"
        # Setup Cj, Cb for Dual Simplex (original objective, or -Z if Min)
        cj_map_dual = {} # Same as cj_map_phase_II from Primal
        for var in var_names_all:
            original_coeff = orig_obj_coeffs_map.get(var, 0.0)
            if var not in orig_obj_coeffs_map and var.startswith('e'): original_coeff = 0.0
            # For Dual Simplex, artificial variables should not be present or should have been eliminated
            # If they are somehow there, their Cj should be highly penalized or 0 if non-problematic.
            # Let's assume they are not an issue for typical Dual Simplex start.
            elif var in artificial_var_names: # This case needs careful thought for Dual Simplex start
                 original_coeff = 0.0 # Or a very large negative M if still in problem.
            cj_map_dual[var] = -original_coeff if was_minimized else original_coeff

        iteration_dual = 0
        while iteration_dual <= MAX_ITERATIONS:
            cb_map_dual = {var_b: cj_map_dual.get(var_b, 0.0) for var_b in current_basis_names}
            tableau = create_hec_tableau(current_A_matrix, current_b_vector, var_names_all,
                                         current_basis_names, cj_map_dual, cb_map_dual)

            # Check initial dual feasibility (Cj-Zj <= 0 for Max)
            if iteration_dual == 0:
                cj_minus_zj_vals = tableau.iloc[-1, 3:-1].astype(float).values
                if not np.all(np.array(cj_minus_zj_vals) <= EPSILON):
                    steps_html += format_tableau_html_hec(tableau, f"Dual Simplex - Iteration {iteration_dual} (Initial)", 
                                                          "Initial tableau is NOT dual feasible (Cj-Zj condition not met for Max). Dual Simplex cannot start.")
                    return steps_html, {"status": "Error", "message": "Not dual feasible to start Dual Simplex."}
            
            pivot_row_idx_in_A, leaving_desc = find_dual_pivot_row_hec(tableau)
            if pivot_row_idx_in_A == -1: # Primal feasible, so optimal
                # ... (Optimal solution found for Dual Simplex)
                final_obj_val_maximized = float(tableau[tableau["_Labels_Zj_CjZj"] == "Zj"].iloc[0]["_bi_Actual"])
                actual_obj_val = -final_obj_val_maximized if was_minimized else final_obj_val_maximized
                solution_vars = {} # Populate
                num_decision_vars = len(orig_obj_coeffs_map)
                for i in range(num_decision_vars): 
                    var_name_orig_decision = f"x{i+1}"
                    if var_name_orig_decision in current_basis_names:
                        row_idx_in_basis = current_basis_names.index(var_name_orig_decision)
                        solution_vars[var_name_orig_decision] = float(current_b_vector[row_idx_in_basis])
                    else: solution_vars[var_name_orig_decision] = 0.0
                steps_html += format_tableau_html_hec(tableau, f"Dual Simplex - Iteration {iteration_dual} (Optimal)", f"{leaving_desc}\nOptimal Solution. Z = {actual_obj_val:.4f}")
                return steps_html, {"status": "Optimal", "value": actual_obj_val, "variables": solution_vars}

            pivot_col_var_idx, entering_desc = find_dual_pivot_column_hec(tableau, pivot_row_idx_in_A)
            if pivot_col_var_idx == -1: # Primal infeasible (Dual unbounded)
                # ... (Primal infeasible message)
                steps_html += format_tableau_html_hec(tableau, f"Dual Simplex - Iteration {iteration_dual}", f"{leaving_desc}\n{entering_desc}\nProblem is Infeasible (Dual Unbounded).")
                return steps_html, {"status": "Infeasible", "message": entering_desc}

            entering_var_name = var_names_all[pivot_col_var_idx]
            leaving_var_name = current_basis_names[pivot_row_idx_in_A]
            combined_desc = leaving_desc + "\n" + entering_desc
            steps_html += format_tableau_html_hec(tableau, f"Dual Simplex - Iteration {iteration_dual}", combined_desc,
                                                  pivot_r_idx_df=pivot_row_idx_in_A, pivot_c_idx_df_internal=pivot_col_var_idx,
                                                  entering_var_name=entering_var_name, leaving_var_name=leaving_var_name)
            
            current_A_matrix, current_b_vector = perform_pivot_operation_hec(current_A_matrix, current_b_vector, pivot_row_idx_in_A, pivot_col_var_idx)
            current_basis_names[pivot_row_idx_in_A] = entering_var_name
            
            iteration_dual += 1
            if iteration_dual > MAX_ITERATIONS: return steps_html + "<p style='color:red'>Dual Simplex: Max iter.</p>", {"status": "Max Iter (Dual Simplex)"}
        # Fallback for Dual path
        return steps_html + "<p style='color:red'>Dual Simplex did not conclude.</p>", {"status": "Error (Dual Simplex)"}

    else:
        return "<p style='color:red'>Error: Unknown solver method selected.</p>", None

# Global variable hack for standardize_problem, to be removed if standardize_problem is refactored
obj_type_ui_global = "Maximize" 