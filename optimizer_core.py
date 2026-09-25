import os

# Disable CrewAI background telemetry calls to eliminate 30-second network timeouts
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

import json
import re
from pathlib import Path
from dotenv import load_dotenv
import gurobipy as gp
from gurobipy import GRB
from json_repair import repair_json
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# Force load .env variables immediately upon import
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ==========================================
# 1. HELPER & UTILITY FUNCTIONS
# ==========================================

def parse_json(raw_text: str) -> dict:
    """Parses JSON safely from LLM output, extracting content within curly braces."""
    if not raw_text or not isinstance(raw_text, str):
        return {}
    start = raw_text.find('{')
    end = raw_text.rfind('}') + 1
    clean_str = raw_text[start:end] if start != -1 and end > start else raw_text.strip()
    try:
        return json.loads(repair_json(clean_str))
    except Exception:
        return {}


def get_llms():
    """Initializes upgraded Gemini 3.x models using CrewAI's native LLM wrapper."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")

    # High-speed model for classification, code generation, and tool execution
    fast = LLM(
        model="gemini/gemini-3.8-flash",
        temperature=0.0,
        api_key=key
    )

    # Low-latency model for rapid synthetic dataset synthesis
    cheap = LLM(
        model="gemini/gemini-3.5-flash-lite",
        temperature=0.1,
        api_key=key
    )

    # High-reasoning model for executive reporting and interactive chatbot Q&A
    reasoning = LLM(
        model="gemini/gemini-3.8-flash",
        temperature=0.2,
        api_key=key
    )

    return fast, cheap, reasoning


# ==========================================
# 2. GUROBI EXECUTION TOOL
# ==========================================

class DynamicGurobiExecutionTool(BaseTool):
    name: str = "Dynamic Gurobi Model Execution Tool"
    description: str = "Executes dynamically generated Gurobi Python code against input data."

    def _run(self, gurobi_code: str, input_data: dict) -> str:
        try:
            scope = {}
            exec(gurobi_code, globals(), scope)
            if "run_model" in scope:
                res = scope["run_model"](input_data)
                return json.dumps(res) if isinstance(res, (dict, list)) else str(res)
            return json.dumps({"status": "ERROR", "details": "Missing run_model(data) function."})
        except Exception as e:
            return json.dumps({"status": "ERROR", "details": f"Dynamic Execution Error: {str(e)}"})


# ==========================================
# 3. WORKFLOW PIPELINE
# ==========================================

def classify_problem(prompt: str) -> str:
    """
    Part 1: Fast problem classification.
    Uses local keyword regex pattern matching for instantaneous responses,
    falling back to a direct Gemini 3.8 Flash API call if no rule matches.
    """
    text = prompt.lower()
    if re.search(r'\b(route|routing|vehicle|fleet|truck|vrp|delivery|tsp|stop)\b', text):
        return "Vehicle Routing & Fleet Scheduling (VRP)"
    elif re.search(r'\b(inventory|stock|safety stock|reorder|holding cost|eoq|multi-echelon)\b', text):
        return "Multi-Echelon Inventory Optimization"
    elif re.search(r'\b(facility|location|open plant|site selection|warehouse location|locate)\b', text):
        return "Facility Location & Capacity Allocation"
    elif re.search(r'\b(production|blend|blending|manufacturing|batch|assembly)\b', text):
        return "Production Planning & Blending"
    elif re.search(r'\b(ship|shipping|transport|flow|lane|origin|destination|supplier|cheapest route)\b', text):
        return "Network Flow / Transportation Problem"

    # Direct Gemini LLM fallback (bypasses CrewAI agent wrapper overhead)
    try:
        fast, _, _ = get_llms()
        raw = fast.call(f"""
        Classify this supply chain scenario request into EXACTLY ONE of these categories:
        - Network Flow / Transportation Problem
        - Facility Location & Capacity Allocation
        - Multi-Echelon Inventory Optimization
        - Vehicle Routing & Fleet Scheduling (VRP)
        - Production Planning & Blending
        
        Scenario: '{prompt}'
        
        Return ONLY raw JSON: {{"problem_class": "Category Name"}}
        """)
        parsed = parse_json(raw)
        return parsed.get("problem_class", "General Logistics Optimization")
    except Exception:
        return "General Logistics Optimization"


def define_schema(prompt: str, problem_class: str) -> str:
    """Part 2: Defines input data schema and parameter requirements."""
    fast, _, _ = get_llms()
    
    agent = Agent(
        role="Schema Architect",
        goal="Define required input data schema and table structures.",
        backstory="Operations Research expert specializing in supply chain data architecture and parameter requirements.",
        llm=fast
    )
    
    task = Task(
        description=f"""
        Problem Category: '{problem_class}'
        Scenario: '{prompt}'
        
        Specify the required input data schema, tables, and parameters needed to model this problem in Gurobi.
        Provide a short markdown summary listing the required entities, capacity constraints, demand targets, and cost fields.
        
        Return JSON object with key "data_requirements_summary".
        """,
        expected_output="Raw JSON object with 'data_requirements_summary'",
        agent=agent
    )
    
    raw = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()
    return parse_json(raw).get("data_requirements_summary", "Standard supply chain network parameters required: suppliers, warehouses, capacities, demands, and costs.")


def generate_synthetic_data(prompt: str, problem_class: str, schema_summary: str) -> dict:
    """Phase 2A: Synthesizes a valid benchmark JSON dataset."""
    _, cheap, _ = get_llms()
    
    agent = Agent(
        role="Data Architect",
        goal="Generate valid benchmark JSON datasets rapidly.",
        backstory="Data engineer specializing in strict supply chain JSON schema compliance and benchmark data generation.",
        llm=cheap
    )
    
    task = Task(
        description=f"""
        Generate a clean, valid benchmark JSON data object for this problem:
        Problem Category: '{problem_class}'
        User Request: '{prompt}'
        Required Schema Summary: '{schema_summary}'
        
        Output ONLY raw valid JSON containing key-value data structures. Ensure double quotes around all keys and strings.
        """,
        expected_output="Raw valid JSON payload",
        agent=agent
    )
    
    raw = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()
    fallback = {
        "suppliers": ["Chicago", "Atlanta"],
        "warehouses": ["Dallas", "New_York"],
        "capacities": {"Chicago": 50.0, "Atlanta": 200.0},
        "demands": {"Dallas": 100.0, "New_York": 80.0},
        "costs": {
            "Chicago_to_Dallas": 10.0, "Chicago_to_New_York": 12.0,
            "Atlanta_to_Dallas": 18.0, "Atlanta_to_New_York": 8.0
        }
    }
    parsed = parse_json(raw)
    return parsed if parsed else fallback


def generate_gurobi_code(prompt: str, problem_class: str, input_json_data: dict) -> str:
    """Phase 2B: Generates executable Gurobi Python code."""
    fast, _, _ = get_llms()
    
    agent = Agent(
        role="Gurobi Architect",
        goal="Write robust gurobipy Python code matching input data schema.",
        backstory="Operations Research modeling specialist building production-ready Gurobi optimization scripts.",
        llm=fast
    )
    
    task = Task(
        description=f"""
        Write a Python script defining a function `run_model(data)` using `gurobipy`.
        
        Problem Scenario: '{prompt}'
        Problem Category: '{problem_class}'
        Sample Input Data Schema Structure: {json.dumps(input_json_data)}
        
        CRITICAL GUROBI CODE RULES FOR `run_model(data)`:
        1. Import gurobipy safely inside or outside: `import gurobipy as gp`, `from gurobipy import GRB`.
        2. Initialize the model explicitly: `m = gp.Model("SupplyChain")`.
        3. DO NOT use variable names `m` or `model` for loops, dictionaries, or data keys to prevent variable shadowing.
        4. Ensure `m.addConstr()` is called directly on the Gurobi model instance `m`.
        5. SELF-HEALING FALLBACK:
           If `m.status == GRB.INFEASIBLE`, compute IIS (`m.computeIIS()`), apply relaxation slacks (`m.feasRelaxS(0, True, False, True)`), and re-optimize (`m.optimize()`). Set status to "RELAXED_OPTIMAL".
        6. MULTI-OBJECTIVE WEIGHTING:
           If `data` contains a 'weights' key, extract cost_weight and carbon_weight to formulate the objective function accordingly.
        7. SAFE SHADOW PRICE EXTRACTION:
           Extract constraint dual values (`c.Pi`) ONLY if the model is continuous (`m.IsMIP == 0`) and solved (`m.SolCount > 0`). Wrap `c.Pi` queries in a `try-except AttributeError` block.
        8. EXACT ROUTE DICTIONARY KEYS:
           Return an optimal routes list formatted strictly with exact key names: 
           "routes": [{{"source": "OriginNode", "target": "DestinationNode", "flow": 100.0}}]
        9. RETURN DICTIONARY STRUCTURE:
           The function `run_model(data)` MUST return a dict with keys:
           - "status": String (e.g., "OPTIMAL", "RELAXED_OPTIMAL", "INFEASIBLE")
           - "cost": Float (Objective value `m.ObjVal` if solution exists, else 0.0)
           - "routes": List of route dicts [{{"source": ..., "target": ..., "flow": ...}}]
           - "shadow_prices": Dict of {{constr_name: dual_val}}
           - "details": String summary of model statistics
        
        Return JSON object with key "gurobi_code".
        """,
        expected_output="Raw JSON object with key 'gurobi_code'",
        agent=agent
    )
    
    raw = str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()
    return parse_json(raw).get("gurobi_code", "")


def execute_optimization_crew(gurobi_code: str, json_data: dict) -> dict:
    """Phase 3: Executes model code via custom tool and synthesizes executive report."""
    fast, _, reasoning = get_llms()
    
    solver_agent = Agent(
        role="Solver Specialist",
        goal="Execute Gurobi optimization tool quickly.",
        backstory="Gurobi execution engine specialist running compiled optimization routines.",
        tools=[DynamicGurobiExecutionTool()],
        llm=fast
    )
    
    executive_agent = Agent(
        role="Executive Strategist",
        goal="Synthesize raw optimization outputs into concise executive briefings.",
        backstory="Supply chain executive advisor translating linear programming results into strategic decision insights.",
        llm=reasoning
    )

    task_solve = Task(
        description=f"Directly call Dynamic Gurobi Execution Tool with gurobi_code: '''{gurobi_code}''' and input_data: {json.dumps(json_data)}.",
        expected_output="JSON string from execution tool.",
        agent=solver_agent
    )
    
    task_brief = Task(
        description="Write a concise 2-3 paragraph executive summary focusing on key operational decisions, cost bottlenecks, and dual value/shadow price insights.",
        expected_output="Markdown executive summary briefing.",
        agent=executive_agent
    )

    optimization_crew = Crew(
        agents=[solver_agent, executive_agent],
        tasks=[task_solve, task_brief],
        process=Process.sequential
    )
    
    res = str(optimization_crew.kickoff()).strip()
    raw_output = task_solve.output.raw if hasattr(task_solve, 'output') else ""
    
    return {
        "executive_report": res,
        "raw_solver_output": raw_output
    }


def run_post_solve_chatbot(query: str, session_context: dict, chat_history: list) -> str:
    """Phase 4: Session Memory Chatbot for post-optimization user Q&A."""
    _, _, reasoning = get_llms()
    
    agent = Agent(
        role="Advisor Chatbot",
        goal="Answer scenario follow-up questions and guide model re-run prompts.",
        backstory="Logistics decision support consultant reviewing model formulations, flow results, and business strategies.",
        llm=reasoning
    )
    
    history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-6:]])

    task = Task(
        description=f"""
        CONTEXT:
        - User Scenario: '{session_context.get('user_prompt')}'
        - Problem Class: '{session_context.get('problem_class')}'
        - Active Input Data: {json.dumps(session_context.get('active_json_data'))}
        - Optimizer Raw Results: {session_context.get('raw_solver_output')}
        - Executive Report: {session_context.get('executive_report')}
        
        CONVERSATION HISTORY:
        {history_str}
        
        NEW USER QUERY: '{query}'
        
        CRITICAL RE-RUN INSTRUCTIONS:
        If the query requires modifying the optimization model, changing data parameters, or introducing new constraints:
        1. Inform the user clearly that their request requires updating the model formulation and re-running the solver.
        2. Provide step-by-step instructions:
           - Step 1: Copy the custom generated prompt provided below.
           - Step 2: Paste it into Section 1 ('Supply Chain Scenario Query') at the top of the app.
           - Step 3: Click 'Classify & Define Schema' to trigger execution.
        3. Provide a synthesized custom prompt block in markdown blockquotes combining the original scenario with their new requirement.
        
        Otherwise, answer the question accurately based on the scenario context provided.
        """,
        expected_output="Clear markdown response addressing the query or providing re-run guidance with a custom prompt block.",
        agent=agent
    )
    
    return str(Crew(agents=[agent], tasks=[task]).kickoff()).strip()