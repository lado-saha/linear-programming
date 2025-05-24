import gradio as gr
import numpy as np
import pandas as pd

EPSILON = 1e-9 # For floating point comparisons

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

    A_matrix_list = []
    b_vector_list = []
    
    # Initial variable names for decision variables
    var_names = [f"x{i+1}" for i in range(num_vars)]
    
    # Counters for new variables
    s_idx_counter = 0 # Slack
    e_idx_counter = 0 # Surplus
    a_idx_counter = 0 # Artificial
    
    # Store types of variables added per constraint to correctly build basis later
    constraint_var_info = [] 

    # First pass: determine counts of slack, surplus, artificial vars
    # And prepare basic A_matrix rows for original vars
    num_slack = 0
    num_surplus = 0
    num_artificial = 0

    temp_A_rows = []
    temp_b_values = []
    temp_ops = []

    for constr in constraints_data:
        coeffs = np.array(constr['coeffs'], dtype=float)
        rhs = float(constr['rhs'])
        op = constr['op']

        if rhs < 0:
            coeffs = -coeffs
            rhs = -rhs
            if op == "<=": op = ">="
            elif op == ">=": op = "<="
        
        temp_A_rows.append(list(coeffs)) # Original variable coefficients
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
    obj_coeffs_std = np.concatenate([obj_coeffs_std, np.zeros(num_slack + num_surplus + num_artificial)])

    # Second pass: build full A_matrix and var_names
    full_A_matrix_list = []
    
    current_s_col_offset = num_vars
    current_e_col_offset = num_vars + num_slack
    current_a_col_offset = num_vars + num_slack + num_surplus

    s_local_idx = 0
    e_local_idx = 0
    a_local_idx = 0

    # Determine basis variable types and names during this pass for get_initial_basis
    # This list will store {'type': 'slack/artificial', 'name': 's1/a1', 'original_constraint_index': i}
    basis_construction_details = []


    for i in range(len(temp_A_rows)):
        original_coeffs = temp_A_rows[i]
        op = temp_ops[i]
        
        # Start with original variable coefficients, then zeros for all new variable types
        row = list(original_coeffs) + [0.0] * (num_slack + num_surplus + num_artificial)
        
        if op == "<=":
            s_idx_counter += 1
            var_name = f"s{s_idx_counter}"
            var_names.append(var_name)
            row[current_s_col_offset + s_local_idx] = 1.0
            basis_construction_details.append({'type': 'slack', 'name': var_name, 'original_constraint_index': i})
            s_local_idx += 1
        elif op == ">=":
            e_idx_counter += 1
            surplus_var_name = f"e{e_idx_counter}"
            var_names.append(surplus_var_name)
            row[current_e_col_offset + e_local_idx] = -1.0 # Subtract surplus
            e_local_idx += 1
            
            a_idx_counter += 1
            artificial_var_name = f"a{a_idx_counter}"
            var_names.append(artificial_var_name)
            row[current_a_col_offset + a_local_idx] = 1.0 # Add artificial
            basis_construction_details.append({'type': 'artificial', 'name': artificial_var_name, 'original_constraint_index': i})
            a_local_idx += 1
        elif op == "=":
            a_idx_counter += 1
            artificial_var_name = f"a{a_idx_counter}"
            var_names.append(artificial_var_name)
            row[current_a_col_offset + a_local_idx] = 1.0 # Add artificial
            basis_construction_details.append({'type': 'artificial', 'name': artificial_var_name, 'original_constraint_index': i})
            a_local_idx += 1
        
        full_A_matrix_list.append(row)
        b_vector_list.append(temp_b_values[i])

    return (
        np.array(full_A_matrix_list, dtype=float),
        np.array(b_vector_list, dtype=float),
        obj_coeffs_std, # Already includes zeros for new vars
        var_names,
        num_vars, # original_num_vars
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
        # The 'name' in basis_construction_details is the name of the slack or artificial var entering basis
        basis_var_names.append(entry['name'])
        try:
            basis_indices.append(all_var_names.index(entry['name']))
        except ValueError:
            # This should not happen if var_names is constructed correctly in standardize_problem
            raise Exception(f"Variable {entry['name']} not found in all_var_names list for basis construction.")

    return basis_indices, basis_var_names


def create_tableau(A, b, obj_coeffs, var_names, basis_indices, phase_one=False, original_obj_coeffs_phase2=None, num_artificial_vars=0):
    num_constraints, num_total_vars_in_A = A.shape
    
    # Ensure obj_coeffs matches the number of variables in A
    if len(obj_coeffs) != num_total_vars_in_A:
        # This can happen if obj_coeffs_std was not padded correctly or var_names used for tableau doesn't match A.
        # For Phase I, obj_coeffs is effectively different.
        # For Phase II, original_obj_coeffs_phase2 should be used and must match.
        if not phase_one and original_obj_coeffs_phase2 is not None and len(original_obj_coeffs_phase2) != num_total_vars_in_A:
             raise ValueError(f"Obj.coeffs length ({len(original_obj_coeffs_phase2)}) mismatch with A columns ({num_total_vars_in_A}) in Phase II.")
        # If phase_one or original_obj_coeffs_phase2 is None, it implies direct use of obj_coeffs.
        elif phase_one and len(obj_coeffs) != num_total_vars_in_A :
             # This state needs careful handling for Phase I's objective
             pass # Phase I objective is constructed below.

    tableau_data = np.zeros((num_constraints + 1, num_total_vars_in_A + 1)) # +1 for Cj-Zj row, +1 for RHS column
    
    tableau_data[:num_constraints, :num_total_vars_in_A] = A
    tableau_data[:num_constraints, -1] = b
    
    current_basis_var_names = [var_names[i] for i in basis_indices]

    # Objective function row (Cj-Zj)
    if phase_one:
        cj_phase1 = np.zeros(num_total_vars_in_A)
        # Artificial variables are typically at the end of the var_names list
        # Their original indices start after decision, slack, and surplus variables.
        # num_artificial_vars helps identify them if var_names includes them grouped.
        
        # More robust: identify by name prefix 'a'
        for i, name in enumerate(var_names):
            if name.startswith('a'):
                cj_phase1[i] = 1.0 # Cost of artificial variable is 1 in Phase I obj (min sum of a_i)
        
        # Cb: Coefficients of basic variables in Phase I objective
        cb_phase1 = np.array([cj_phase1[idx] for idx in basis_indices])
        
        zj_phase1 = cb_phase1 @ A
        tableau_data[-1, :num_total_vars_in_A] = cj_phase1 - zj_phase1
        tableau_data[-1, -1] = -(cb_phase1 @ b) # Initial W value (objective function value for phase 1)
    else:
        # Phase II objective or direct solve
        active_obj_coeffs = original_obj_coeffs_phase2 if original_obj_coeffs_phase2 is not None else obj_coeffs
        if len(active_obj_coeffs) != num_total_vars_in_A:
             raise ValueError(f"Phase II obj_coeffs length {len(active_obj_coeffs)} != A columns {num_total_vars_in_A}")

        cb = np.array([active_obj_coeffs[idx] for idx in basis_indices])
        zj = cb @ A
        tableau_data[-1, :num_total_vars_in_A] = active_obj_coeffs - zj
        tableau_data[-1, -1] = -(cb @ b) # Initial Z value

    df_cols = var_names + ["RHS"]
    df_index = current_basis_var_names + ["Cj-Zj"]
    tableau_df = pd.DataFrame(tableau_data, columns=df_cols, index=df_index)
    
    return tableau_df


def find_pivot_column(tableau_df, phase_one=False): # Maximize is implied for phase II, phase I is specific
    cj_zj_row = tableau_df.iloc[-1, :-1].values.astype(float) # Exclude RHS, ensure float for comparison
    
    if phase_one: # Minimizing sum of artificials (W). Cj-Zj for W is (Cj_art - Zj_art). Enter most positive.
        if np.all(cj_zj_row <= EPSILON): return -1 # Optimal for Phase I (all Cj-Zj <= 0 for min problem form)
        pivot_col_idx = np.argmax(cj_zj_row)
        if cj_zj_row[pivot_col_idx] <= EPSILON : return -1 
    else: # Phase II (Maximizing original Z). Enter most positive Cj-Zj.
          # If problem was Min Z, obj_coeffs were negated, so we still look for most positive Cj-Zj.
        if np.all(cj_zj_row <= EPSILON): return -1 # Optimal (all Cj-Zj <= 0 for max problem)
        pivot_col_idx = np.argmax(cj_zj_row) # Max value in Cj-Zj for entering var
        if cj_zj_row[pivot_col_idx] <= EPSILON : return -1 # Optimal if max is non-positive
            
    return pivot_col_idx # This is np.intp or similar, cast to int when used for df.columns[idx]

def find_pivot_row(tableau_df, pivot_col_idx):
    # pivot_col_idx is already an integer index for the column
    pivot_column_values = tableau_df.iloc[:-1, pivot_col_idx].values.astype(float) # Exclude Cj-Zj row
    rhs_values = tableau_df.iloc[:-1, -1].values.astype(float)
    
    ratios = []
    valid_row_indices = [] # Stores original indices in the tableau_df (0 to num_constraints-1)
    
    for i in range(len(pivot_column_values)):
        if pivot_column_values[i] > EPSILON: # Denominator must be positive
            ratios.append(rhs_values[i] / pivot_column_values[i])
            valid_row_indices.append(i)
    
    if not valid_row_indices: # All pivot column entries are <= 0
        return -1, "Unbounded: All pivot column elements are non-positive."

    min_ratio = float('inf')
    pivot_row_idx = -1
    
    # Find minimum positive ratio
    for i, original_idx in enumerate(valid_row_indices):
        current_ratio = ratios[i] # Use pre-calculated ratio
        if current_ratio < min_ratio - EPSILON : 
            min_ratio = current_ratio
            pivot_row_idx = original_idx
        elif abs(current_ratio - min_ratio) < EPSILON: 
            # Tie-breaking: Bland's rule - choose leaving variable with smallest subscript.
            # Here, we'd compare tableau_df.index[original_idx] vs tableau_df.index[pivot_row_idx]
            # For simplicity, current code takes the first one encountered or one with smaller row index.
            # Let's refine to pick smaller index if names are 's1', 'x2' etc.
            if pivot_row_idx == -1 or original_idx < pivot_row_idx: # Simpler: pick smaller row index
                 pivot_row_idx = original_idx

    if pivot_row_idx == -1: # Should be caught by 'not valid_row_indices'
        return -1, "Error in ratio test, no valid pivot row found."

    # Prepare ratio description string
    ratio_strings = []
    for i in range(len(pivot_column_values)):
        desc_val = pivot_column_values[i]
        if desc_val > EPSILON:
            ratio_strings.append(f"{tableau_df.index[i]}: {rhs_values[i]:.2f}/{desc_val:.2f} = {rhs_values[i]/desc_val:.2f}")
        else:
            ratio_strings.append(f"{tableau_df.index[i]}: N/A (coeff ≤ 0)")
    
    min_ratio_text = f"Min Ratio = {min_ratio:.2f} (for row {tableau_df.index[pivot_row_idx]})"
    ratio_test_desc = "Ratio Test (RHS / Pivot Column Value):\n" + "\n".join(ratio_strings) + "\n" + min_ratio_text
    
    return pivot_row_idx, ratio_test_desc # pivot_row_idx is standard Python int

def perform_pivot_operation(tableau_df, pivot_row_idx, pivot_col_idx):
    # pivot_row_idx is int, pivot_col_idx needs to be int for .columns access
    pivot_col_idx_int = int(pivot_col_idx) 

    new_tableau_df = tableau_df.copy()
    pivot_element = new_tableau_df.iloc[pivot_row_idx, pivot_col_idx_int]
    
    if abs(pivot_element) < EPSILON:
        raise ValueError("Pivot element is zero, cannot perform pivot operation.")

    # Normalize pivot row
    new_tableau_df.iloc[pivot_row_idx, :] /= pivot_element
    
    # Update other rows (including Cj-Zj)
    for i in range(new_tableau_df.shape[0]):
        if i != pivot_row_idx:
            factor = new_tableau_df.iloc[i, pivot_col_idx_int]
            new_tableau_df.iloc[i, :] -= factor * new_tableau_df.iloc[pivot_row_idx, :]
            
    # Update basis variable name
    entering_var_name = new_tableau_df.columns[pivot_col_idx_int]
    new_basis_var_names = list(new_tableau_df.index)
    new_basis_var_names[pivot_row_idx] = entering_var_name
    new_tableau_df.index = new_basis_var_names
    
    return new_tableau_df


def format_tableau_html(df, iteration_str, description, pivot_r_idx=None, pivot_c_idx=None, entering_var=None, leaving_var=None):
    
    def style_pivot(data, pivot_r_idx, pivot_c_idx_as_int, entering_var, leaving_var):
        s = pd.DataFrame('', index=data.index, columns=data.columns)
        if pivot_r_idx is not None and pivot_c_idx_as_int is not None:
            # Style pivot row (excluding RHS, and if pivot_r_idx is not Cj-Zj row)
            if pivot_r_idx < len(data.index) - 1:
                for col_idx in range(len(data.columns) - 1): 
                    s.iloc[pivot_r_idx, col_idx] = 'background-color: lightyellow;'
            
            # Style pivot column (excluding Cj-Zj row, and if pivot_c_idx is not RHS column)
            if pivot_c_idx_as_int < len(data.columns) - 1:
                for row_idx in range(len(data.index) - 1):
                    current_style = s.iloc[row_idx, pivot_c_idx_as_int]
                    # If cell is also in pivot row, lightyellow is already set. Add border or change to combined style.
                    # For simplicity, lightblue might override or combine depending on CSS.
                    # Let's make it a distinct style for column that doesn't assume prior style.
                    s.iloc[row_idx, pivot_c_idx_as_int] = (current_style + ';' if current_style else '') + 'background-color: lightblue;'


            # Override pivot element itself to be distinctly yellow and bold
            s.iloc[pivot_r_idx, pivot_c_idx_as_int] = 'background-color: yellow; font-weight: bold; border: 1.5px solid red;'
        return s

    # Ensure pivot_c_idx is int for the styler if provided
    pivot_c_idx_int = int(pivot_c_idx) if pivot_c_idx is not None else None

    styled_df = df.style.set_table_attributes('class="dataframe table table-striped table-hover table-sm" border="1"') \
                        .format("{:.2f}")
    
    if pivot_r_idx is not None and pivot_c_idx_int is not None : # Use the int version
         styled_df = styled_df.apply(style_pivot, axis=None, 
                                     pivot_r_idx=pivot_r_idx, pivot_c_idx_as_int=pivot_c_idx_int,
                                     entering_var=entering_var, leaving_var=leaving_var)

    html_table = styled_df.to_html(escape=False)
    
    header = f"<h3>{iteration_str}</h3>" # Use iteration_str directly
    if entering_var and leaving_var:
        header += f"<p><b>Entering Variable:</b> {entering_var}, <b>Leaving Variable:</b> {leaving_var}</p>"
    if description:
         header += f"<p>{description.replace(chr(10), '<br>')}</p>" # Replace newline for HTML
    
    return f"{header}{html_table}<hr>"

def solve_simplex_problem(obj_type, obj_coeffs_str, constraints_inputs, num_vars_val, num_constraints_val):
    steps_html = ""
    MAX_ITERATIONS = 50

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
            parsed_constraints.append({'coeffs': coeffs, 'op': op, 'rhs': float(rhs_str)})
    except ValueError as e:
        return f"<p style='color:red'>Error: Invalid numeric input for coefficients or RHS: {e}</p>", None
    except Exception as e:
        return f"<p style='color:red'>Error parsing inputs: {e}</p>", None

    A_std, b_std, obj_coeffs_for_tableau, var_names_std, orig_num_vars, num_s, num_sur, num_a, was_minimized, basis_constr_details = \
        standardize_problem(obj_type, obj_coeffs_list, parsed_constraints, num_vars_val)
    
    steps_html += "<h2>Initial Problem Setup</h2>"
    # ... (initial problem display - can be enhanced)
    steps_html += f"<p><b>Variables after standardization:</b> {', '.join(var_names_std)}</p><hr>"

    # --- Phase I (if needed) ---
    if num_a > 0:
        steps_html += "<h2>Phase I: Minimize Sum of Artificial Variables</h2>"
        # Get initial basis for Phase I (typically artificial vars, or slack if constraint was already equality-like)
        phase_I_basis_indices, _ = get_initial_basis(basis_constr_details, var_names_std)
        
        # Create Phase I tableau
        tableau_df = create_tableau(A_std, b_std, obj_coeffs_for_tableau, var_names_std, 
                                    phase_I_basis_indices, phase_one=True, num_artificial_vars=num_a)
        
        iteration = 0
        steps_html += format_tableau_html(tableau_df, f"Phase I - Iteration {iteration}", "Initial Phase I Tableau")

        for iteration_ph1 in range(1, MAX_ITERATIONS + 1):
            pivot_col_idx = find_pivot_column(tableau_df, phase_one=True)

            if pivot_col_idx == -1: # Phase I optimal
                phase_one_obj_val = -float(tableau_df.iloc[-1, -1]) # W = - (value in Cj-Zj, RHS)
                if abs(phase_one_obj_val) > EPSILON:
                    msg = f"Phase I ends. Sum of artificial variables W = {phase_one_obj_val:.4f} > 0. Problem is Infeasible."
                    steps_html += f"<p style='color:red;font-weight:bold;'>{msg}</p>"
                    return steps_html, {"status": "Infeasible", "message": msg}
                else:
                    steps_html += "<p style='color:green;font-weight:bold;'>Phase I ends. W = 0. Feasible solution found. Proceeding to Phase II.</p>"
                    # Prepare for Phase II
                    # Drop artificial variable columns if they are non-basic (value 0 and not in basis)
                    # Current tableau_df already has artificial variables potentially in basis (with value 0)
                    # Reconstruct Cj-Zj for original objective
                    
                    current_basis_vars_final_ph1 = list(tableau_df.index[:-1])
                    current_basis_indices_final_ph1 = [var_names_std.index(var) for var in current_basis_vars_final_ph1]
                    
                    A_for_phase2 = tableau_df.iloc[:-1, :-1].values # Current A matrix
                    b_for_phase2 = tableau_df.iloc[:-1, -1].values   # Current b vector
                    
                    # Use original objective coefficients (obj_coeffs_for_tableau was already adjusted for min/max and padded)
                    tableau_df = create_tableau(A_for_phase2, b_for_phase2, obj_coeffs_for_tableau, 
                                                list(tableau_df.columns[:-1]), # current var names in tableau
                                                current_basis_indices_final_ph1,
                                                phase_one=False, original_obj_coeffs_phase2=obj_coeffs_for_tableau,
                                                num_artificial_vars=num_a) # num_a for reference if needed by create_tableau
                    break 
            
            pivot_row_idx, ratio_desc = find_pivot_row(tableau_df, int(pivot_col_idx)) # Cast pivot_col_idx
            
            if pivot_row_idx == -1:
                msg = "Phase I: Unbounded (or error in pivot row selection)."
                steps_html += f"<p style='color:orange;font-weight:bold;'>{msg}</p>"
                return steps_html, {"status": "Error in Phase I", "message": msg}

            entering_var = tableau_df.columns[int(pivot_col_idx)] # Cast
            leaving_var = tableau_df.index[pivot_row_idx]
            
            tableau_df = perform_pivot_operation(tableau_df, pivot_row_idx, pivot_col_idx) # pivot_col_idx (np.intp) handled inside
            steps_html += format_tableau_html(tableau_df, f"Phase I - Iteration {iteration_ph1}", ratio_desc, pivot_row_idx, int(pivot_col_idx), entering_var, leaving_var)
            if iteration_ph1 == MAX_ITERATIONS :
                 return steps_html + "<p style='color:red;font-weight:bold;'>Phase I: Exceeded maximum iterations.</p>", {"status": "Max Iterations in Phase I"}
        else: # Loop finished without break (max iterations)
             if iteration_ph1 >= MAX_ITERATIONS : # Check if loop completed due to max_iterations
                return steps_html + "<p style='color:red;font-weight:bold;'>Phase I: Exceeded maximum iterations.</p>", {"status": "Max Iterations in Phase I"}


    else: # No artificial variables
        steps_html += "<h2>Simplex Method (No Phase I needed)</h2>"
        initial_basis_indices, _ = get_initial_basis(basis_constr_details, var_names_std)
        tableau_df = create_tableau(A_std, b_std, obj_coeffs_for_tableau, var_names_std, initial_basis_indices, phase_one=False)

    # --- Phase II (or direct solution) ---
    phase_name = "Phase II" if num_a > 0 else "Simplex Solution"
    steps_html += f"<h2>{phase_name}</h2>"
    iteration = 0 
    steps_html += format_tableau_html(tableau_df, f"{phase_name} - Iteration {iteration}", f"Initial Tableau for {phase_name}")

    for iteration_ph2 in range(1, MAX_ITERATIONS + 1):
        pivot_col_idx = find_pivot_column(tableau_df, phase_one=False)

        if pivot_col_idx == -1: # Optimal solution found
            obj_val_final = -float(tableau_df.iloc[-1, -1]) 
            if was_minimized:
                obj_val_final = -obj_val_final

            solution = {"status": "Optimal", "value": obj_val_final, "variables": {}}
            for i in range(orig_num_vars): 
                var_name_orig = f"x{i+1}"
                if var_name_orig in tableau_df.index: # If basic
                    solution["variables"][var_name_orig] = tableau_df.loc[var_name_orig, "RHS"]
                else: # If non-basic
                    solution["variables"][var_name_orig] = 0.0
            
            steps_html += f"<p style='color:green;font-weight:bold;'>{phase_name} ends. Optimal solution found.</p>"
            return steps_html, solution

        pivot_row_idx, ratio_desc = find_pivot_row(tableau_df, int(pivot_col_idx)) # Cast pivot_col_idx

        if pivot_row_idx == -1: # Unbounded solution
            msg = f"{phase_name} ends. Problem is Unbounded. ({ratio_desc})"
            steps_html += f"<p style='color:orange;font-weight:bold;'>{msg}</p>"
            return steps_html, {"status": "Unbounded", "message": msg}
            
        entering_var = tableau_df.columns[int(pivot_col_idx)] # Cast
        leaving_var = tableau_df.index[pivot_row_idx]
        
        tableau_df = perform_pivot_operation(tableau_df, pivot_row_idx, pivot_col_idx) # pivot_col_idx (np.intp) handled inside
        steps_html += format_tableau_html(tableau_df, f"{phase_name} - Iteration {iteration_ph2}", ratio_desc, pivot_row_idx, int(pivot_col_idx), entering_var, leaving_var)
    
    # If loop finishes, max iterations exceeded for Phase II
    return steps_html + f"<p style='color:red;font-weight:bold;'>{phase_name}: Exceeded maximum iterations.</p>", {"status": f"Max Iterations in {phase_name}"}


# --- Gradio UI ---
css = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.gr-interface { /* background-color: #f9f9f9; */ } /* Let theme handle it */
/* .dark .gr-interface { background-color: #2b2b2b; } */ /* Let theme handle it */
.dataframe { margin-bottom: 20px; font-size: 0.9em; border-collapse: collapse; width: auto; }
.dataframe th, .dataframe td { border: 1px solid #ddd; padding: 6px; text-align: center; }
.dark .dataframe th, .dark .dataframe td { border: 1px solid #555; }
.dataframe th { background-color: #f0f0f0; }
.dark .dataframe th { background-color: #3a3a3a; }
.output_html_class table { width: auto !important; margin: 1em auto !important; } 
.output_html_class hr { margin-top: 20px; margin-bottom: 20px; border-top: 1px solid #ccc; }
.output_html_class h2 { margin-top: 1.5em; }
.output_html_class h3 { margin-top: 1em; }
"""

MAX_CONSTRAINTS = 10 

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# Simplex Method Solver for Linear Programming")
    gr.Markdown("Enter the linear programming problem details below. Ensure all inputs are valid numbers.")

    with gr.Row():
        obj_type = gr.Radio(["Maximize", "Minimize"], label="Objective Type", value="Maximize")
        num_vars = gr.Number(label="Number of Decision Variables (e.g., x1, x2)", value=2, minimum=1, step=1, precision=0)
    
    obj_coeffs = gr.Textbox(label="Objective Function Coefficients (comma-separated, e.g., 3,5 for 3x1 + 5x2)", placeholder="e.g., 3,5", value="3,5")
    num_constraints = gr.Number(label="Number of Constraints", value=1, minimum=1, maximum=MAX_CONSTRAINTS, step=1, precision=0)

    constraint_rows_ui_components = []
    for i in range(MAX_CONSTRAINTS):
        with gr.Group(visible=(i < int(num_constraints.value))) as const_group: # Initial visibility
            gr.Markdown(f"**Constraint {i+1}**")
            with gr.Row():
                coeffs_inp = gr.Textbox(label=f"LHS Coefficients (comma-separated)", placeholder="e.g., 2,1", scale=3)
                op_inp = gr.Dropdown(["<=", ">=", "="], label="Operator", value="<=", scale=1)
                rhs_inp = gr.Textbox(label="RHS Value", placeholder="e.g., 10", scale=1) 
            constraint_rows_ui_components.append({
                'group': const_group,
                'coeffs': coeffs_inp,
                'op': op_inp,
                'rhs': rhs_inp
            })

    # Example default values for first constraint if num_vars=2, num_constraints=1
    if MAX_CONSTRAINTS > 0 and len(constraint_rows_ui_components) > 0 :
        constraint_rows_ui_components[0]['coeffs'].value = "1,1" # Default for constraint 1
        constraint_rows_ui_components[0]['rhs'].value = "100"    # Default for constraint 1


    def update_constraint_rows_visibility(k_str):
        k = int(k_str)
        updates = []
        for i in range(MAX_CONSTRAINTS):
            updates.append(gr.update(visible=(i < k)))
        return updates # Gradio expects a list of updates matching the number of outputs

    num_constraints.change(
        fn=update_constraint_rows_visibility,
        inputs=num_constraints,
        outputs=[row['group'] for row in constraint_rows_ui_components] 
    )

    solve_button = gr.Button("Solve Problem", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("## Solution Steps and Result")
    
    output_steps_html = gr.HTML(label="Simplex Tableaux and Steps", elem_classes="output_html_class") # Added elem_classes
    output_result_text = gr.Textbox(label="Final Result", lines=5, interactive=False)

    def assemble_and_solve_problem(obj_type_val, obj_coeffs_val, num_vars_val, num_constraints_val, 
                                   *flat_constraint_inputs_args):
        
        num_vars_int = int(num_vars_val)
        num_constraints_int = int(num_constraints_val)
        
        constraints_data_dict = {}
        expected_len_flat_args = num_constraints_int * 3 # Each visible constraint contributes 3 fields
        
        actual_flat_args = flat_constraint_inputs_args[:expected_len_flat_args]

        for i in range(num_constraints_int):
            coeffs_val = actual_flat_args[i*3]
            op_val = actual_flat_args[i*3 + 1]
            rhs_val = actual_flat_args[i*3 + 2]
            
            if not coeffs_val or not op_val or not rhs_val: # Basic validation for empty inputs
                 return "<p style='color:red'>Error: One or more constraint fields are empty.</p>", "Error: Empty constraint field."

            constraints_data_dict[f"const_{i}_coeffs"] = coeffs_val
            constraints_data_dict[f"const_{i}_op"] = op_val
            constraints_data_dict[f"const_{i}_rhs"] = rhs_val

        html_output_content, solution_data = solve_simplex_problem(
            obj_type_val, obj_coeffs_val, constraints_data_dict, num_vars_int, num_constraints_int
        )
        
        result_summary_text = ""
        if solution_data:
            status = solution_data.get("status", "Unknown")
            result_summary_text = f"Status: {status}\n"
            if status == "Optimal":
                result_summary_text += f"Optimal Objective Value: {solution_data.get('value', 'N/A'):.4f}\nVariables:\n"
                for var, val in solution_data.get("variables", {}).items():
                    result_summary_text += f"  {var} = {val:.4f}\n"
            elif "message" in solution_data:
                 result_summary_text += f"Message: {solution_data['message']}"
            elif status == "Max Iterations in Phase I" or status == "Max Iterations in Phase II" or status.startswith("Max Iterations"):
                result_summary_text += "The solver exceeded the maximum number of iterations."
            # else:
            #    result_summary_text += "Could not determine final solution details."


        return html_output_content, result_summary_text

    # Prepare list of all potential input components for solve_button.click
    all_individual_constraint_fields_for_button = []
    for i in range(MAX_CONSTRAINTS):
        all_individual_constraint_fields_for_button.extend([
            constraint_rows_ui_components[i]['coeffs'],
            constraint_rows_ui_components[i]['op'],
            constraint_rows_ui_components[i]['rhs']
        ])

    solve_button.click(
        fn=assemble_and_solve_problem,
        inputs=[obj_type, obj_coeffs, num_vars, num_constraints] + all_individual_constraint_fields_for_button,
        outputs=[output_steps_html, output_result_text]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
