import gradio as gr
# Import the HEC-style solver function
from simplex_logic import solve_simplex_problem_hec_style

MAX_CONSTRAINTS = 10

# No custom CSS should be needed beyond what theme provides for table rendering by pandas.to_html
with gr.Blocks(theme=gr.themes.Ocean()) as demo:

    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align:center;">
                <h1 style="color: inherit; margin-bottom: 2px;">Simplex & Big M Method Solver</h1>
                <p style="color: inherit; font-size: 1.1rem;">Linear Programming Problem Resolution</p>
            </div>
            """
        )
    gr.Markdown("---")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=280):  # Slightly wider min_width
            with gr.Group():
                gr.Markdown(
                    "### <div style='text-align:center;'>Problem Definition</div>")
                obj_type = gr.Radio(["Maximize", "Minimize"], label="Objective Type",
                                    value="Maximize", info="Select problem type.")
                with gr.Row():
                    num_vars = gr.Number(label="Variables", value=2, minimum=1, step=1,
                                         precision=0, scale=1, info="No. of decision variables.")
                    num_constraints = gr.Number(label="Constraints", value=1, minimum=1,
                                                maximum=MAX_CONSTRAINTS, step=1, precision=0, scale=1, info="No. of constraints.")
                obj_coeffs = gr.Textbox(label="Objective Function Coefficients",
                                        placeholder="e.g., 3,5", value="3,5", info="Comma-separated.")
                with gr.Accordion("Define Constraints Details", open=True):
                    constraint_rows_ui_components = []
                    for i in range(MAX_CONSTRAINTS):
                        is_visible = (i < int(num_constraints.value))
                        with gr.Group(visible=is_visible) as const_group:
                            gr.Markdown(f"**Constraint {i+1}**")
                            with gr.Row():
                                coeffs_inp = gr.Textbox(
                                    label=f"LHS Coeffs", placeholder="e.g., 2,1", scale=3)
                                op_inp = gr.Dropdown(
                                    ["<=", ">=", "="], label="Op", value="<=", scale=1)
                                rhs_inp = gr.Textbox(
                                    label="RHS", placeholder="e.g., 10", scale=1)
                            constraint_rows_ui_components.append(
                                {'group': const_group, 'coeffs': coeffs_inp, 'op': op_inp, 'rhs': rhs_inp})
                if MAX_CONSTRAINTS > 0:  # Set default for the difficult problem
                    constraint_rows_ui_components[0]['coeffs'].value = "1,3"
                    constraint_rows_ui_components[0]['op'].value = "<="
                    constraint_rows_ui_components[0]['rhs'].value = "15"
                    if MAX_CONSTRAINTS > 1:  # For constraint 2
                        constraint_rows_ui_components[1]['coeffs'].value = "2,1"
                        constraint_rows_ui_components[1]['op'].value = ">="
                        constraint_rows_ui_components[1]['rhs'].value = "8"
                    if MAX_CONSTRAINTS > 2:  # For constraint 3
                        constraint_rows_ui_components[2]['coeffs'].value = "1,1"
                        constraint_rows_ui_components[2]['op'].value = "="
                        constraint_rows_ui_components[2]['rhs'].value = "7"
                    # If default is 1 constraint but we have defaults for 3
                    if int(num_constraints.value) < 3 and MAX_CONSTRAINTS >= 3:
                        # Trigger visibility update if possible, or user sets it.
                        num_constraints.value = 1
                        # num_constraints = gr.Number(label="Constraints", value=3, ...) # If you want to hardcode for test

            solve_button = gr.Button(
                "📊 Solve Problem", variant="primary", size="lg")

        with gr.Column(scale=7):
            with gr.Tabs():
                with gr.TabItem("Solution Steps & Tableaux"):
                    gr.Markdown(
                        "### <div style='text-align:center;'>Step-by-Step Solution</div>")
                    output_steps_html = gr.HTML(label=None)
                with gr.TabItem("Final Summary"):
                    gr.Markdown(
                        "### <div style='text-align:center;'>Result Summary</div>")
                    output_result_text = gr.Textbox(
                        label=None, lines=8, interactive=False, show_copy_button=True)

    def update_constraint_rows_visibility(k_str):
        try:
            k = int(k_str)
        except ValueError:
            k = 1
        updates = []
        for i in range(MAX_CONSTRAINTS):
            updates.append(gr.update(visible=(i < k)))
        return updates

    output_groups_for_visibility_update = [
        row['group'] for row in constraint_rows_ui_components]
    num_constraints.change(
        fn=update_constraint_rows_visibility,
        inputs=num_constraints,
        outputs=output_groups_for_visibility_update
    )

    def assemble_and_solve_problem_wrapper(obj_type_val, obj_coeffs_val, num_vars_val, num_constraints_val,
                                           *flat_constraint_inputs_args):
        try:
            num_vars_int = int(num_vars_val)
            num_constraints_int = int(num_constraints_val)
        except ValueError:
            return "<p style='color:red; text-align:center;'>Error: Number of variables or constraints is not a valid integer.</p>", \
                   "Input Error: Number of variables or constraints must be a valid integer."
        constraints_data_dict = {}
        for i in range(num_constraints_int):
            idx_offset = i * 3
            coeffs_val = flat_constraint_inputs_args[idx_offset]
            op_val = flat_constraint_inputs_args[idx_offset + 1]
            rhs_val = flat_constraint_inputs_args[idx_offset + 2]
            if not isinstance(coeffs_val, str) or not isinstance(op_val, str) or not isinstance(rhs_val, str) or \
               coeffs_val.strip() == "" or rhs_val.strip() == "":
                return "<p style='color:red; text-align:center;'>Error: One or more constraint fields are empty or invalid.</p>", \
                       "Input Error: Active constraint fields are empty or invalid."
            constraints_data_dict[f"const_{i}_coeffs"] = coeffs_val
            constraints_data_dict[f"const_{i}_op"] = op_val
            constraints_data_dict[f"const_{i}_rhs"] = rhs_val

        # Call the HEC-style solver
        html_output_content, solution_data = solve_simplex_problem_hec_style(
            obj_type_val, obj_coeffs_val, constraints_data_dict, num_vars_int, num_constraints_int
        )
        result_summary_text = ""
        if solution_data:
            status = solution_data.get("status", "Unknown")
            result_summary_text = f"Status: {status}\n"
            if status == "Optimal":
                obj_value = solution_data.get('value', 'N/A')
                result_summary_text += f"Optimal Objective Value: {obj_value}\nVariables:\n"
                for var, val in solution_data.get("variables", {}).items():
                    result_summary_text += f"  {var} = {val}\n"
            elif "message" in solution_data:
                result_summary_text += f"Message: {solution_data['message']}"
            elif "Max Iterations" in status:
                result_summary_text += "Solver exceeded max iterations."
            elif status != "Unknown":
                result_summary_text += "Solver finished with an unhandled status."
        else:
            if html_output_content and "Error:" in html_output_content:
                result_summary_text = "Error in processing. See steps."
            else:
                result_summary_text = "Unexpected error, no details."
        return html_output_content, result_summary_text

    all_individual_constraint_fields_for_button = []
    for i in range(MAX_CONSTRAINTS):
        all_individual_constraint_fields_for_button.extend([
            constraint_rows_ui_components[i]['coeffs'],
            constraint_rows_ui_components[i]['op'],
            constraint_rows_ui_components[i]['rhs']
        ])

    # Set default num_constraints to 3 for the test case
    # This is a bit of a hack to pre-fill for the difficult problem.
    # A cleaner way would be example buttons or a more robust state update.
    demo.load(lambda: gr.update(value=3), None, num_constraints)

    solve_button.click(
        fn=assemble_and_solve_problem_wrapper,
        inputs=[obj_type, obj_coeffs, num_vars, num_constraints] +
        all_individual_constraint_fields_for_button,
        outputs=[output_steps_html, output_result_text]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
