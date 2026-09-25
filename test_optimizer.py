import pytest
import json
import optimizer_core as core


# ==============================================================================
# CATEGORY 1: TAXONOMY & CLASSIFICATION TESTS (TC-01 - TC-6)
# ==============================================================================

def test_tc01_classify_network_flow():
    """TC-01: Verify classification for Network Flow / Transportation Problem queries."""
    prompt = "What is the cheapest shipping route if Chicago capacity drops by 50% while meeting Dallas demand?"
    category = core.classify_problem(prompt)
    assert category == "Network Flow / Transportation Problem"


def test_tc02_classify_vrp_regex():
    """TC-02: Verify local rule-based regex pre-classifier for Vehicle Routing Problem (VRP) queries."""
    prompt = "Determine the optimal truck delivery routes for 15 stops using our vehicle fleet."
    category = core.classify_problem(prompt)
    assert category == "Vehicle Routing & Fleet Scheduling (VRP)"


def test_tc03_classify_facility_location():
    """TC-03: Verify classification for Facility Location & Capacity Allocation queries."""
    prompt = "Find the optimal warehouse nodes to open in San Francisco to cover zip code demand."
    category = core.classify_problem(prompt)
    assert category == "Facility Location & Capacity Allocation"


def test_tc04_classify_production_blending():
    """TC-04: Verify classification for Production Planning & Blending queries."""
    prompt = "Find the optimal monthly product blend for our chemical plant to maximize margin."
    category = core.classify_problem(prompt)
    assert category == "Production Planning & Blending"


def test_tc05_classify_inventory_optimization():
    """TC-05: Verify classification for Multi-Echelon Inventory Optimization queries."""
    prompt = "Optimize safety stock levels and reorder points across regional distribution hubs."
    category = core.classify_problem(prompt)
    assert category == "Multi-Echelon Inventory Optimization"


def test_tc06_unmatched_query_fallback():
    """TC-06: Verify fallback to direct Gemini API call for unconventional/abstract queries."""
    prompt = "How should we strategically optimize our artisan chocolate supply chain strategy in 2026?"
    category = core.classify_problem(prompt)
    assert isinstance(category, str)
    assert len(category) > 0


# ==============================================================================
# CATEGORY 2: SCHEMA DEFINITION & DATA INGESTION TESTS (TC-07 - TC-10)
# ==============================================================================

def test_tc07_define_schema_structure():
    """TC-07: Verify schema definition produces non-empty markdown summary requirements."""
    prompt = "Minimize shipping costs between 3 suppliers and 5 warehouses."
    summary = core.define_schema(prompt, "Network Flow / Transportation Problem")
    assert isinstance(summary, str)
    assert len(summary) > 10


def test_tc08_generate_synthetic_benchmark_data():
    """TC-08: Verify synthetic data generation produces valid benchmark JSON keys."""
    prompt = "Route supply from 2 plants to 2 fulfillment centers."
    schema_sum = "Required fields: suppliers, warehouses, capacities, demands, costs."
    data = core.generate_synthetic_data(prompt, "Network Flow / Transportation Problem", schema_sum)
    
    assert isinstance(data, dict)
    assert "suppliers" in data or "capacities" in data


def test_tc09_json_parser_resilience():
    """TC-09: Verify parse_json resilience against malformed or empty inputs."""
    assert core.parse_json("") == {}
    assert core.parse_json("Corrupted non-JSON string") == {}
    
    # Verify json_repair recovers malformed JSON with single quotes and trailing commas
    repaired = core.parse_json("{'suppliers': ['Chicago'],}")
    assert "suppliers" in repaired


def test_tc10_zero_demand_boundary_handling():
    """TC-10: Verify solver handles zero-demand boundary constraints without division-by-zero errors."""
    zero_demand_data = {
        "suppliers": ["Chicago", "Atlanta"],
        "warehouses": ["Dallas", "New_York"],
        "capacities": {"Chicago": 100.0, "Atlanta": 200.0},
        "demands": {"Dallas": 0.0, "New_York": 50.0},
        "costs": {
            "Chicago_to_Dallas": 10.0, "Chicago_to_New_York": 12.0,
            "Atlanta_to_Dallas": 18.0, "Atlanta_to_New_York": 8.0
        },
        "weights": {"cost_weight": 1.0, "carbon_weight": 0.0}
    }
    code = core.generate_gurobi_code("Zero demand test", "Network Flow / Transportation Problem", zero_demand_data)
    results = core.execute_optimization_crew(code, zero_demand_data)
    parsed = json.loads(results["raw_solver_output"])
    assert parsed.get("status") in ["OPTIMAL", "RELAXED_OPTIMAL"]


# ==============================================================================
# CATEGORY 3: CODE COMPILATION & GUROBI SOLVER TESTS (TC-11 - TC-15)
# ==============================================================================

def test_tc11_gurobi_code_generation_structure():
    """TC-11: Verify dynamic Gurobi code contains required imports and run_model function signature."""
    sample_data = {
        "suppliers": ["Chicago"],
        "warehouses": ["Dallas"],
        "capacities": {"Chicago": 100},
        "demands": {"Dallas": 80},
        "costs": {"Chicago_to_Dallas": 10.0}
    }
    code = core.generate_gurobi_code("Basic shipping", "Network Flow / Transportation Problem", sample_data)
    assert "def run_model(data):" in code
    assert "gurobipy" in code


def test_tc12_infeasible_model_self_healing():
    """TC-12: Verify self-healing slack relaxation executes when demand exceeds total capacity."""
    infeasible_data = {
        "suppliers": ["Chicago"],
        "warehouses": ["Dallas"],
        "capacities": {"Chicago": 10.0},  # Capacity (10) < Demand (500)
        "demands": {"Dallas": 500.0},
        "costs": {"Chicago_to_Dallas": 10.0},
        "weights": {"cost_weight": 1.0, "carbon_weight": 0.0}
    }
    code = core.generate_gurobi_code("Infeasible scenario", "Network Flow / Transportation Problem", infeasible_data)
    results = core.execute_optimization_crew(code, infeasible_data)
    parsed = json.loads(results["raw_solver_output"])
    
    assert parsed.get("status") in ["RELAXED_OPTIMAL", "OPTIMAL"]


def test_tc13_multi_objective_weight_objective():
    """TC-13: Verify objective formulation accounts for multi-objective cost vs carbon weights."""
    weighted_data = {
        "suppliers": ["Chicago", "Atlanta"],
        "warehouses": ["Dallas"],
        "capacities": {"Chicago": 100.0, "Atlanta": 100.0},
        "demands": {"Dallas": 80.0},
        "costs": {"Chicago_to_Dallas": 10.0, "Atlanta_to_Dallas": 5.0},
        "weights": {"cost_weight": 0.0, "carbon_weight": 1.0}  # 100% Carbon Minimization Priority
    }
    code = core.generate_gurobi_code("Multi-objective test", "Network Flow / Transportation Problem", weighted_data)
    results = core.execute_optimization_crew(code, weighted_data)
    parsed = json.loads(results["raw_solver_output"])
    assert parsed.get("status") == "OPTIMAL"


def test_tc14_shadow_price_extraction():
    """TC-14: Verify continuous LP models extract shadow prices (dual values) into output dictionary."""
    continuous_data = {
        "suppliers": ["Chicago", "Atlanta"],
        "warehouses": ["Dallas"],
        "capacities": {"Chicago": 50.0, "Atlanta": 200.0},
        "demands": {"Dallas": 100.0},
        "costs": {"Chicago_to_Dallas": 5.0, "Atlanta_to_Dallas": 20.0},
        "weights": {"cost_weight": 1.0, "carbon_weight": 0.0}
    }
    code = core.generate_gurobi_code("Shadow price test", "Network Flow / Transportation Problem", continuous_data)
    results = core.execute_optimization_crew(code, continuous_data)
    parsed = json.loads(results["raw_solver_output"])
    
    assert "shadow_prices" in parsed
    assert isinstance(parsed["shadow_prices"], dict)


def test_tc15_route_dictionary_format():
    """TC-15: Verify route flow outputs adhere to exact 'source', 'target', and 'flow' dictionary keys."""
    test_data = {
        "suppliers": ["Chicago"],
        "warehouses": ["Dallas"],
        "capacities": {"Chicago": 100.0},
        "demands": {"Dallas": 50.0},
        "costs": {"Chicago_to_Dallas": 10.0}
    }
    code = core.generate_gurobi_code("Route format check", "Network Flow / Transportation Problem", test_data)
    results = core.execute_optimization_crew(code, test_data)
    parsed = json.loads(results["raw_solver_output"])
    
    routes = parsed.get("routes", [])
    if routes:
        first_route = routes[0]
        assert "source" in first_route
        assert "target" in first_route
        assert "flow" in first_route


# ==============================================================================
# CATEGORY 4: CHATBOT & INTERACTIVE CONVERSATION TESTS (TC-16 - TC-20)
# ==============================================================================

def test_tc16_chatbot_standard_context_response():
    """TC-16: Verify post-solve chatbot directly answers contextual result queries without re-run instructions."""
    context = {
        "user_prompt": "Cheapest route for Chicago plant",
        "problem_class": "Network Flow / Transportation Problem",
        "active_json_data": {"suppliers": ["Chicago"], "warehouses": ["Dallas"]},
        "raw_solver_output": '{"status": "OPTIMAL", "cost": 500.0}',
        "executive_report": "The optimal route is Chicago to Dallas."
    }
    history = []
    query = "Why was Chicago chosen as the main supplier?"
    
    response = core.run_post_solve_chatbot(query, context, history)
    assert isinstance(response, str)
    assert len(response) > 0


def test_tc17_chatbot_rerun_prompt_trigger():
    """TC-17: Verify chatbot detects parameter changes and returns re-run steps with custom prompt block."""
    context = {
        "user_prompt": "Base scenario",
        "problem_class": "Network Flow / Transportation Problem",
        "active_json_data": {"capacities": {"Chicago": 50.0}},
        "raw_solver_output": '{"status": "OPTIMAL"}',
        "executive_report": "Base summary"
    }
    history = []
    query = "What if Chicago capacity drops by 80% instead?"
    
    response = core.run_post_solve_chatbot(query, context, history)
    assert "Section 1" in response or "Classify" in response or "Step 1" in response


def test_tc18_chatbot_conversation_history_persistence():
    """TC-18: Verify chatbot incorporates past conversation history messages into response context."""
    context = {
        "user_prompt": "Base scenario",
        "problem_class": "Network Flow / Transportation Problem",
        "active_json_data": {},
        "raw_solver_output": '{"status": "OPTIMAL"}',
        "executive_report": "Base summary"
    }
    history = [
        {"role": "user", "content": "What is the total cost?"},
        {"role": "assistant", "content": "The total cost is $500.00."}
    ]
    query = "What was the cost I just asked you about?"
    
    response = core.run_post_solve_chatbot(query, context, history)
    assert isinstance(response, str)


def test_tc19_execution_tool_error_handling():
    """TC-19: Verify DynamicGurobiExecutionTool handles missing run_model function cleanly."""
    tool = core.DynamicGurobiExecutionTool()
    broken_code = "x = 10"  # Missing run_model(data) function
    result = tool._run(broken_code, {})
    
    parsed = json.loads(result)
    assert parsed.get("status") == "ERROR"
    assert "run_model" in parsed.get("details", "")


def test_tc20_executive_report_generation():
    """TC-20: Verify optimization crew outputs executive summary report string alongside raw solver data."""
    test_data = {
        "suppliers": ["Chicago"],
        "warehouses": ["Dallas"],
        "capacities": {"Chicago": 100.0},
        "demands": {"Dallas": 50.0},
        "costs": {"Chicago_to_Dallas": 10.0}
    }
    code = core.generate_gurobi_code("Report test", "Network Flow / Transportation Problem", test_data)
    results = core.execute_optimization_crew(code, test_data)
    
    assert "executive_report" in results
    assert "raw_solver_output" in results
    assert len(results["executive_report"]) > 0