import streamlit as st
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import optimizer_core as core

st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")
st.title("📦 Supply Chain Decision Control Agent")

# Session State Initialization
for key in ['schema_design', 'gurobi_code', 'active_json_data', 'execution_results', 'chat_history']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'chat_history' else []

if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = "What is the cheapest shipping route if Chicago plant capacity drops by 50% while meeting Dallas demand?"


def render_network_graph(routes: list):
    """Renders NetworkX graph visualizer."""
    if not routes:
        st.warning("No active flow routes to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    G = nx.DiGraph()

    for item in routes:
        if isinstance(item, dict):
            src = item.get('source') or item.get('from') or item.get('origin')
            tgt = item.get('target') or item.get('to') or item.get('destination')
            flow = item.get('flow') or item.get('volume') or 0.0
            if src and tgt:
                G.add_edge(str(src), str(tgt), weight=float(flow))

    if G.number_of_edges() == 0:
        st.warning("No valid edges found.")
        return

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='#1f77b4', ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='#ff7f0e', arrowsize=15, ax=ax)
    
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    plt.title("Logistics Flow Map", fontsize=11, fontweight='bold')
    plt.axis('off')
    st.pyplot(fig)


# SECTION 1: SCENARIO QUERY
st.subheader("1. Scenario Query")
user_prompt_input = st.text_area("Enter your Scenario Prompt:", value=st.session_state.user_prompt, height=80)

col1, col2 = st.columns([1.5, 4])
with col1:
    if st.button("Classify & Define Schema"):
        st.session_state.user_prompt = user_prompt_input
        st.session_state.schema_design = None
        st.session_state.gurobi_code = None
        st.session_state.active_json_data = None
        st.session_state.execution_results = None
        st.session_state.chat_history = []

        with st.spinner("Analyzing scenario..."):
            prob_class = core.classify_problem(user_prompt_input)
            schema_sum = core.define_schema(user_prompt_input, prob_class)
            st.session_state.schema_design = {"problem_class": prob_class, "data_requirements_summary": schema_sum}
            st.rerun()

with col2:
    if st.button("Reset Session"):
        for key in ['schema_design', 'gurobi_code', 'active_json_data', 'execution_results']:
            st.session_state[key] = None
        st.session_state.chat_history = []
        st.rerun()

st.markdown("---")

# SECTION 2: DATA INGESTION & MODEL COMPILATION
if st.session_state.schema_design:
    prob_class = st.session_state.schema_design.get("problem_class")
    st.success(f"🏷️ **Classified Category:** {prob_class}")
    st.markdown("**Required Parameters:**")
    st.markdown(st.session_state.schema_design.get("data_requirements_summary"))

    cost_weight = st.slider("Priority: Financial Cost Minimization", 0.0, 1.0, 0.8, 0.1)
    weights_dict = {"cost_weight": cost_weight, "carbon_weight": round(1.0 - cost_weight, 2)}

    data_opt = st.radio("Select Data Source:", ["Generate Synthetic Data", "Upload CSV/Excel", "Upload Custom JSON"])

    if data_opt == "Generate Synthetic Data":
        if st.button("Generate & Compile Model"):
            with st.spinner("Generating data & model..."):
                json_data = core.generate_synthetic_data(st.session_state.user_prompt, prob_class, st.session_state.schema_design.get("data_requirements_summary"))
                json_data["weights"] = weights_dict
                st.session_state.active_json_data = json_data
                st.session_state.gurobi_code = core.generate_gurobi_code(st.session_state.user_prompt, prob_class, json_data)
                st.session_state.execution_results = None
                st.rerun()

    elif data_opt == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file and st.button("Process & Compile Model"):
            try:
                if uploaded_file.name.endswith('.csv'):
                    converted = pd.read_csv(uploaded_file).to_dict(orient="records")
                else:
                    converted = {sheet: frame.to_dict(orient="records") for sheet, frame in pd.read_excel(uploaded_file, sheet_name=None).items()}
                payload = {"uploaded_data": converted, "weights": weights_dict}
                st.session_state.active_json_data = payload
                st.session_state.gurobi_code = core.generate_gurobi_code(st.session_state.user_prompt, prob_class, payload)
                st.session_state.execution_results = None
                st.rerun()
            except Exception as e:
                st.error(f"Upload error: {e}")

    else:
        uploaded_json = st.file_uploader("Upload JSON", type=["json"])
        if uploaded_json and st.button("Apply JSON & Compile Model"):
            try:
                parsed = json.load(uploaded_json)
                parsed["weights"] = weights_dict
                st.session_state.active_json_data = parsed
                st.session_state.gurobi_code = core.generate_gurobi_code(st.session_state.user_prompt, prob_class, parsed)
                st.session_state.execution_results = None
                st.rerun()
            except Exception as e:
                st.error(f"JSON Error: {e}")

    if st.session_state.active_json_data and st.session_state.gurobi_code:
        c1, c2 = st.columns(2)
        with c1:
            with st.expander("View Input JSON", expanded=False):
                st.json(st.session_state.active_json_data)
        with c2:
            with st.expander("View Gurobi Code", expanded=False):
                st.code(st.session_state.gurobi_code, language="python")

        st.markdown("---")

        # SECTION 3: SOLVER & INSIGHTS
        st.subheader("3. Optimization Solver & Insights")
        if st.button("🚀 Run Optimizer"):
            with st.spinner("Solving model..."):
                st.session_state.execution_results = core.execute_optimization_crew(st.session_state.gurobi_code, st.session_state.active_json_data)
                st.rerun()

        if st.session_state.execution_results:
            results = st.session_state.execution_results
            raw_str = results.get("raw_solver_output", "")
            clean_str = raw_str[raw_str.find('{'):raw_str.rfind('}')+1] if '{' in raw_str else "{}"
            parsed_solver = json.loads(clean_str) if clean_str != "{}" else {}
            routes = parsed_solver.get("routes", [])

            st.markdown("#### Part 1: Solver Execution")
            st.info(f"Status: {parsed_solver.get('status', 'OPTIMAL')}")

            st.markdown("#### Part 2: Visual Insights")
            col_v1, col_v2 = st.columns([3, 2])
            with col_v1:
                render_network_graph(routes)
            with col_v2:
                if routes:
                    st.dataframe(pd.DataFrame(routes), use_container_width=True)

            st.markdown("#### Part 3: Executive Summary & Chatbot")
            st.markdown(results["executive_report"])

            st.markdown("---")
            st.subheader("💬 Interactive Chatbot")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if user_query := st.chat_input("Ask a question about the results..."):
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                context_payload = {
                    "user_prompt": st.session_state.user_prompt,
                    "problem_class": prob_class,
                    "active_json_data": st.session_state.active_json_data,
                    "raw_solver_output": results.get("raw_solver_output"),
                    "executive_report": results.get("executive_report")
                }
                response = core.run_post_solve_chatbot(user_query, context_payload, st.session_state.chat_history)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})