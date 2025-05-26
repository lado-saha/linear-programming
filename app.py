import gradio as gr
#import all dependenciess
# Import the main solver orchestrator
from simplex_logic import solve_simplex_main

MAX_CONSTRAINTS = 10

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    with gr.Row():
        gr.Markdown(  # ... Title Markdown ...
            """
            <div style="text-align:center;">
                <h1 style="color: inherit; margin-bottom: 2px;">Simplex, Big M & Dual Solver</h1>
                <p style="color: inherit; font-size: 1.1rem;">Linear Programming Problem Resolution</p>
            </div>
            """
        )
    gr.Markdown("---")

    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=380):
            with gr.Group():
                gr.Markdown(
                    "### <div style='text-align:center;'>Problem Definition</div>")

                # ADD METHOD SELECTION
                method_type_gr = gr.Radio(
                    ["Primal Simplex (Two-Phase)", "Dual Simplex"],
                    label="Solver Method",
                    value="Primal Simplex (Two-Phase)",
                    info="Choose the Simplex method variant."
                )

                obj_type_gr = gr.Radio(["Maximize", "Minimize"], label="Objective Type",
                                       value="Maximize", info="Select problem type.")
                # ... (rest of inputs: num_vars, num_constraints, obj_coeffs, constraints accordion) ...
                with gr.Row():
                    num_vars_gr = gr.Number(label="Variables", value=2, minimum=1, step=1,
                                            precision=0, scale=1, info="No. of decision variables.")
                    num_constraints_gr = gr.Number(label="Constraints", value=1, minimum=1,
                                                   maximum=MAX_CONSTRAINTS, step=1, precision=0, scale=1, info="No. of constraints.")
                obj_coeffs_gr = gr.Textbox(label="Objective Function Coefficients",
                                           placeholder="e.g., 3,5", value="3,5", info="Comma-separated.")
                with gr.Accordion("Define Constraints Details", open=True):
                    constraint_rows_ui_components = []
                    # ... (constraint input loop as before) ...
                    for i in range(MAX_CONSTRAINTS):
                        is_visible = (i < int(num_constraints_gr.value))
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

                # Default values for the "difficult problem" suitable for Primal/Two-Phase
                if MAX_CONSTRAINTS > 0:
                    constraint_rows_ui_components[0]['coeffs'].value = "1,3"
                    constraint_rows_ui_components[0]['op'].value = "<="
                    constraint_rows_ui_components[0]['rhs'].value = "15"
                    if MAX_CONSTRAINTS > 1:
                        constraint_rows_ui_components[1]['coeffs'].value = "2,1"
                        constraint_rows_ui_components[1]['op'].value = ">="
                        constraint_rows_ui_components[1]['rhs'].value = "8"
                    if MAX_CONSTRAINTS > 2:
                        constraint_rows_ui_components[2]['coeffs'].value = "1,1"
                        constraint_rows_ui_components[2]['op'].value = "="
                        constraint_rows_ui_components[2]['rhs'].value = "7"

            solve_button_gr = gr.Button(
                "📊 Solve Problem", variant="primary", size="lg")

        with gr.Column(scale=7):
            with gr.Tabs():
                # ... (Output tabs as before) ...
                with gr.TabItem("Solution Steps & Tableaux"):
                    gr.Markdown(
                        "### <div style='text-align:center;'>Step-by-Step Solution</div>")
                    output_steps_html_gr = gr.HTML(label=None)
                with gr.TabItem("Final Summary"):
                    gr.Markdown(
                        "### <div style='text-align:center;'>Result Summary</div>")
                    output_result_text_gr = gr.Textbox(
                        label=None, lines=8, interactive=False, show_copy_button=True)

    # --- Event Handlers ---
    def update_constraint_rows_visibility(k_str):
        # ... (Same as before) ...
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
    num_constraints_gr.change(
        fn=update_constraint_rows_visibility,
        inputs=num_constraints_gr,
        outputs=output_groups_for_visibility_update
    )

    def assemble_and_solve_problem_wrapper(method_type_val, obj_type_val, obj_coeffs_val,
                                           num_vars_val, num_constraints_val,
                                           *flat_constraint_inputs_args):
        # ... (Input parsing for num_vars, num_constraints, constraints_data_dict as before) ...
        try:
            num_vars_int = int(num_vars_val)
            num_constraints_int = int(num_constraints_val)
        except ValueError:  # ... error handling ...
            return "<p style='color:red; text-align:center;'>Error: Vars/constraints not valid int.</p>", "Input Error: Vars/constraints count invalid."
        constraints_data_dict = {}  # ... populate ...
        for i in range(num_constraints_int):
            idx_offset = i * 3
            coeffs_val = flat_constraint_inputs_args[idx_offset]
            op_val = flat_constraint_inputs_args[idx_offset + 1]
            rhs_val = flat_constraint_inputs_args[idx_offset + 2]
            if not isinstance(coeffs_val, str) or not isinstance(op_val, str) or not isinstance(rhs_val, str) or \
               coeffs_val.strip() == "" or rhs_val.strip() == "":
                return "<p style='color:red; text-align:center;'>Error: Constraint fields empty/invalid.</p>", "Input Error: Active constraint fields empty/invalid."
            constraints_data_dict[f"const_{i}_coeffs"] = coeffs_val
            constraints_data_dict[f"const_{i}_op"] = op_val
            constraints_data_dict[f"const_{i}_rhs"] = rhs_val

        # Call the main solver orchestrator with the selected method
        html_output_content, solution_data = solve_simplex_main(
            obj_type_val, obj_coeffs_val, constraints_data_dict,
            num_vars_int, num_constraints_int, method_type_val  # Pass method_type_val
        )
        # ... (Result formatting as before) ...
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

    all_gr_inputs_for_solver = [
        method_type_gr, obj_type_gr, obj_coeffs_gr, num_vars_gr, num_constraints_gr
    ] + [item for row in constraint_rows_ui_components
         for item in [row['coeffs'], row['op'], row['rhs']]]

    # Initial load to set num_constraints to 3 for the example
    demo.load(lambda: gr.update(value=3), None, num_constraints_gr)

    solve_button_gr.click(
        fn=assemble_and_solve_problem_wrapper,
        inputs=all_gr_inputs_for_solver,
        outputs=[output_steps_html_gr, output_result_text_gr]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
