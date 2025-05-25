import gradio as gr
from simplex_logic import solve_simplex_problem

MAX_CONSTRAINTS = 10

# Using gr.themes.Ocean()
# No custom CSS for layout, but theme provides styling
with gr.Blocks(theme=gr.themes.Ocean()) as demo:

    # --- "Top Bar" Area ---
    with gr.Row():  # This row will act as a header area
        gr.Markdown(
            """
            <div style="text-align:center; padding: 4px 0;">
                <h1 style="color: inherit; margin-bottom: 5px;">Linear Programming Problem Resolutionr</h1>
                <p style="color: inherit; font-size: 1.1em; margin-top:0px;">Simplex and M*</p>
            </div>
            """
        )  # Using Markdown with a bit of inline HTML for better centering and styling control if needed

    gr.Markdown("---")  # Visual separator

    # --- Main Content Area (Two Columns) ---
    # `equal_height=False` can sometimes help with column behavior
    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=350):
            with gr.Group():  # Group to visually contain inputs
                gr.Markdown(
                    "### <div style='text-align:center;'>Problem Definition</div>", elem_id="input_panel_header")

                obj_type = gr.Radio(["Maximize", "Minimize"],
                                    label="Objective Type", value="Maximize", info="Select if you want to maximize or minimize the objective function.")

                with gr.Row():
                    num_vars = gr.Number(label="Variables", value=2, minimum=1, step=1, precision=0,
                                         scale=1, info="Number of decision variables (e.g., x1, x2).")
                    num_constraints = gr.Number(label="Constraints", value=1, minimum=1, maximum=MAX_CONSTRAINTS,
                                                step=1, precision=0, scale=1, info="Number of constraints.")

                obj_coeffs = gr.Textbox(
                    label="Objective Function Coefficients",
                    placeholder="e.g., 3,5 for 3x1 + 5x2", value="3,5",
                    info="Comma-separated coefficients.")

                # Constraints within an Accordion for better organization
                with gr.Accordion("Define Constraints Details", open=True):
                    constraint_rows_ui_components = []
                    for i in range(MAX_CONSTRAINTS):
                        is_visible = (i < int(num_constraints.value))
                        with gr.Group(visible=is_visible) as const_group:
                            gr.Markdown(f"**Constraint {i+1}**")
                            with gr.Row():  # Row for each constraint's components
                                coeffs_inp = gr.Textbox(
                                    label=f"LHS Coeffs", placeholder="e.g., 2,1", scale=3)
                                op_inp = gr.Dropdown(
                                    # Shorter label
                                    ["<=", ">=", "="], label="Op", value="<=", scale=1)
                                rhs_inp = gr.Textbox(
                                    label="RHS", placeholder="e.g., 10", scale=1)  # Shorter label
                            constraint_rows_ui_components.append({
                                'group': const_group,
                                'coeffs': coeffs_inp,
                                'op': op_inp,
                                'rhs': rhs_inp
                            })

                if MAX_CONSTRAINTS > 0 and len(constraint_rows_ui_components) > 0:
                    # Example default
                    constraint_rows_ui_components[0]['coeffs'].value = "1,1"
                    # Example default
                    constraint_rows_ui_components[0]['rhs'].value = "100"

            solve_button = gr.Button(
                "📊 Solve Problem", variant="primary", size="lg")

        # --- Right Column (Outputs) ---
        with gr.Column(scale=7):
            with gr.Tabs():  # Using Tabs for potentially different views of results later
                with gr.TabItem("Solution Steps & Tableaux"):
                    gr.Markdown(
                        "### <div style='text-align:center;'>Step-by-Step Solution</div>")
                    # Label can be omitted if header is present
                    output_steps_html = gr.HTML(label=None)

                with gr.TabItem("Final Summary"):
                    gr.Markdown(
                        "### <div style='text-align:center;'>Result Summary</div>")
                    output_result_text = gr.Textbox(
                        label=None, lines=8, interactive=False, show_copy_button=True)

    # --- Event Handlers ---

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

    def assemble_and_solve_problem(obj_type_val, obj_coeffs_val, num_vars_val, num_constraints_val,
                                   *flat_constraint_inputs_args):
        try:
            num_vars_int = int(num_vars_val)
            num_constraints_int = int(num_constraints_val)
        except ValueError:
            # For HTML output, use Markdown compatible error. For Textbox, plain text.
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
                return "<p style='color:red; text-align:center;'>Error: One or more constraint fields are empty or invalid for currently active constraints.</p>", \
                       "Input Error: One or more active constraint fields are empty or invalid."

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
                obj_value = solution_data.get('value', 'N/A')
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
            # Consider adding an else for other unhandled statuses from simplex_logic
            elif status != "Unknown":  # If status is set but not Optimal or known error
                result_summary_text += "Solver finished with an unhandled status."

        else:  # solution_data is None
            if html_output_content and "Error:" in html_output_content:
                # Error message is already in html_output_content for the HTML display
                result_summary_text = "An error occurred during input processing or problem setup. See steps panel."
            else:  # Should not happen if html_output_content always has error for None solution_data
                result_summary_text = "An unexpected error occurred, and no detailed steps are available."

        # The `solve_button.click` expects two outputs.
        # Both output_steps_html and output_result_text will be updated.
        return html_output_content, result_summary_text

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
        # Both outputs are now under tabs, but Gradio handles updating them correctly
        outputs=[output_steps_html, output_result_text]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
