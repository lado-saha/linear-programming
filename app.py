import gradio as gr
# No need for numpy or pandas here directly, simplex_logic handles them
# EPSILON is not directly used by app.py
from simplex_logic import solve_simplex_problem

# --- Gradio UI ---
css = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
/* .gr-interface { background-color: #f9f9f9; } Let theme handle it */
/* .dark .gr-interface { background-color: #2b2b2b; } Let theme handle it */
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

# Changed theme to Soft for better default dark/light
with gr.Blocks(theme=gr.themes.Ocean(), ) as demo:
    gr.Markdown("# Simplex Method Solver for Linear Programming")
    gr.Markdown(
        "Enter the linear programming problem details below. Ensure all inputs are valid numbers.")

    with gr.Row():
        obj_type = gr.Radio(["Maximize", "Minimize"],
                            label="Objective Type", value="Maximize")
        num_vars = gr.Number(label="Number of Decision Variables (e.g., x1, x2)",
                             value=2, minimum=1, step=1, precision=0)

    obj_coeffs = gr.Textbox(
        label="Objective Function Coefficients (comma-separated, e.g., 3,5 for 3x1 + 5x2)", placeholder="e.g., 3,5", value="3,5")
    num_constraints = gr.Number(label="Number of Constraints", value=1,
                                minimum=1, maximum=MAX_CONSTRAINTS, step=1, precision=0)

    constraint_rows_ui_components = []
    for i in range(MAX_CONSTRAINTS):
        # Initial visibility based on default num_constraints.value
        is_visible = (i < int(num_constraints.value))
        with gr.Group(visible=is_visible) as const_group:
            gr.Markdown(f"**Constraint {i+1}**")
            with gr.Row():
                coeffs_inp = gr.Textbox(
                    label=f"LHS Coefficients (comma-separated)", placeholder="e.g., 2,1", scale=3)
                op_inp = gr.Dropdown(
                    ["<=", ">=", "="], label="Operator", value="<=", scale=1)
                rhs_inp = gr.Textbox(
                    label="RHS Value", placeholder="e.g., 10", scale=1)
            constraint_rows_ui_components.append({
                'group': const_group,
                'coeffs': coeffs_inp,
                'op': op_inp,
                'rhs': rhs_inp
            })

    # Example default values for first constraint if num_vars=2, num_constraints=1
    if MAX_CONSTRAINTS > 0 and len(constraint_rows_ui_components) > 0:
        constraint_rows_ui_components[0]['coeffs'].value = "1,1"
        constraint_rows_ui_components[0]['rhs'].value = "100"

    def update_constraint_rows_visibility(k_str):
        try:
            k = int(k_str)
        except ValueError:  # Handle case where k_str might not be a valid int string temporarily
            k = 1  # Default to 1 if conversion fails
        updates = []
        for i in range(MAX_CONSTRAINTS):
            updates.append(gr.update(visible=(i < k)))
        return updates

    num_constraints.change(
        fn=update_constraint_rows_visibility,
        inputs=num_constraints,
        outputs=[row['group'] for row in constraint_rows_ui_components]
    )

    solve_button = gr.Button("Solve Problem", variant="primary")

    gr.Markdown("---")
    gr.Markdown("## Solution Steps and Result")

    output_steps_html = gr.HTML(label="Simplex Tableaux and Steps",
                                elem_classes="output_html_class")
    output_result_text = gr.Textbox(
        label="Final Result", lines=5, interactive=False)

    def assemble_and_solve_problem(obj_type_val, obj_coeffs_val, num_vars_val, num_constraints_val,
                                   *flat_constraint_inputs_args):
        try:
            num_vars_int = int(num_vars_val)
            num_constraints_int = int(num_constraints_val)
        except ValueError:
            return "<p style='color:red'>Error: Number of variables or constraints is not a valid integer.</p>", "Error: Invalid input for var/constraint count."

        constraints_data_dict = {}
        # Each visible constraint contributes 3 fields: coeffs, op, rhs
        # The *flat_constraint_inputs_args will contain ALL MAX_CONSTRAINTS * 3 inputs
        # We only need to process the ones up to num_constraints_int

        for i in range(num_constraints_int):
            idx_offset = i * 3
            coeffs_val = flat_constraint_inputs_args[idx_offset]
            op_val = flat_constraint_inputs_args[idx_offset + 1]
            rhs_val = flat_constraint_inputs_args[idx_offset + 2]

            if not isinstance(coeffs_val, str) or not isinstance(op_val, str) or not isinstance(rhs_val, str) or \
               coeffs_val.strip() == "" or rhs_val.strip() == "":
                return "<p style='color:red'>Error: One or more constraint fields are empty or invalid for currently active constraints.</p>", "Error: Empty or invalid constraint field."

            constraints_data_dict[f"const_{i}_coeffs"] = coeffs_val
            constraints_data_dict[f"const_{i}_op"] = op_val
            constraints_data_dict[f"const_{i}_rhs"] = rhs_val

        # Call the imported business logic function
        html_output_content, solution_data = solve_simplex_problem(
            obj_type_val, obj_coeffs_val, constraints_data_dict, num_vars_int, num_constraints_int
        )

        result_summary_text = ""
        if solution_data:
            status = solution_data.get("status", "Unknown")
            result_summary_text = f"Status: {status}\n"
            if status == "Optimal":
                obj_value = solution_data.get('value', 'N/A')
                # Ensure obj_value is formatted correctly, handling potential non-numeric N/A
                if isinstance(obj_value, (int, float)):
                    result_summary_text += f"Optimal Objective Value: {obj_value:.4f}\nVariables:\n"
                else:
                    result_summary_text += f"Optimal Objective Value: {obj_value}\nVariables:\n"

                for var, val in solution_data.get("variables", {}).items():
                    if isinstance(val, (int, float)):
                        result_summary_text += f"  {var} = {val:.4f}\n"
                    else:
                        result_summary_text += f"  {var} = {val}\n"

            elif "message" in solution_data:
                result_summary_text += f"Message: {solution_data['message']}"
            elif "Max Iterations" in status:
                result_summary_text += "The solver exceeded the maximum number of iterations."
            # else: (Handle other statuses or default message if needed)
            #    result_summary_text += "Could not determine final solution details."
        else:  # solution_data itself is None, implies error during input parsing in solve_simplex_problem
            if html_output_content and "Error:" in html_output_content:  # The error message is in html_output_content
                result_summary_text = "Error occurred. See steps for details."
            else:
                result_summary_text = "An unexpected error occurred."

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
        inputs=[obj_type, obj_coeffs, num_vars, num_constraints] +
        all_individual_constraint_fields_for_button,
        outputs=[output_steps_html, output_result_text]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
