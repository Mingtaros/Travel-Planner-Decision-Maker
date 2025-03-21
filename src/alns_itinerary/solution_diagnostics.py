import logging
import json
import os
from datetime import datetime
import logging
from data.transport_utils import get_transport_matrix, get_all_locations
from problem.itinerary_problem import TravelItineraryProblem

def setup_logging():
    """
    Set up logging configuration
    """
    # Create logs directory if it doesn't exist
    os.makedirs("log", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"log/alns_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

def detailed_solution_diagnosis(problem, solution):
    """
    Perform a comprehensive diagnosis of solution infeasibility
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Solution vector to diagnose
    
    Returns:
        dict: Detailed diagnostic information
    """
    # Evaluate the solution
    evaluation = problem.evaluate_solution(solution)
    
    # Prepare diagnostic report
    diagnosis = {
        "overall_feasibility": evaluation['is_feasible'],
        "total_cost": evaluation.get('total_cost', 0),
        "budget": problem.budget,
        "constraint_violations": {
            "inequality_violations": [],
            "equality_violations": []
        },
        "summary": {}
    }
    
    # Categorize and analyze violations
    violation_categories = {}
    
    # Process inequality violations
    for violation in evaluation.get('inequality_violations', []):
        category = violation.get('type', 'unknown')
        
        # Count and track violations
        if category not in violation_categories:
            violation_categories[category] = 1
        else:
            violation_categories[category] += 1
        
        # Add detailed violation information
        diagnosis['constraint_violations']['inequality_violations'].append(violation)
    
    # Process equality violations
    for violation in evaluation.get('equality_violations', []):
        category = violation.get('type', 'unknown')
        
        # Count and track violations
        if category not in violation_categories:
            violation_categories[category] = 1
        else:
            violation_categories[category] += 1
        
        # Add detailed violation information
        diagnosis['constraint_violations']['equality_violations'].append(violation)
    
    # Create summary of violation categories
    diagnosis['summary'] = {
        "total_violations": len(diagnosis['constraint_violations']['inequality_violations']) + 
                            len(diagnosis['constraint_violations']['equality_violations']),
        "violation_categories": violation_categories,
        "budget_status": {
            "total_cost": evaluation.get('total_cost', 0),
            "budget_limit": problem.budget,
            "over_budget": evaluation.get('total_cost', 0) > problem.budget
        }
    }
    
    # Detailed logging
    logging.getLogger(__name__).warning(
        f"Solution Diagnosis:\n{json.dumps(diagnosis, indent=2)}"
    )
    
    return diagnosis

def print_solution_diagnostics(problem, solution):
    """
    Print a human-readable diagnostic report
    
    Args:
        problem: TravelItineraryProblem instance
        solution: Solution vector to diagnose
    """
    diagnosis = detailed_solution_diagnosis(problem, solution)
    
    print("\n===== SOLUTION FEASIBILITY DIAGNOSIS =====")
    print(f"Overall Feasibility: {'✖ NOT FEASIBLE' if not diagnosis['overall_feasibility'] else '✓ FEASIBLE'}")
    print(f"Total Violations: {diagnosis['summary']['total_violations']}")
    
    print("\n--- Violation Categories ---")
    for category, count in diagnosis['summary']['violation_categories'].items():
        print(f"{category}: {count} violations")
    
    print("\n--- Budget Analysis ---")
    budget_status = diagnosis['summary']['budget_status']
    print(f"Total Cost: ${budget_status['total_cost']:.2f}")
    print(f"Budget Limit: ${budget_status['budget_limit']:.2f}")
    print(f"Over Budget: {'Yes' if budget_status['over_budget'] else 'No'}")
    
    print("\n--- Detailed Violations ---")
    for vtype, violations in diagnosis['constraint_violations'].items():
        if violations:
            print(f"\n{vtype.replace('_', ' ').title()}:")
            for violation in violations:
                print(f"  - {json.dumps(violation, indent=2)}")
    
    print("\n==========================================")

def debug_solution_generation(problem):
    """
    Debug the solution generation process
    
    Args:
        problem: TravelItineraryProblem instance
    
    Returns:
        List of solutions with their feasibility status
    """
    from alns.alns_core import ALNS
    import numpy as np
    
    solutions = []
    
    # 1. Initial Solution
    alns = ALNS(problem)
    initial_solution = alns.create_initial_solution()
    initial_eval = problem.evaluate_solution(initial_solution)
    
    solutions.append({
        "type": "Initial Solution",
        "solution": initial_solution,
        "is_feasible": initial_eval['is_feasible']
    })
    
    # 2. Perform multiple ALNS runs
    for run in range(3):
        # Reset seed for reproducibility
        np.random.seed(42 + run)
        
        # Run ALNS
        alns = ALNS(problem, seed=42 + run)
        results = alns.run(verbose=False)
        
        solutions.append({
            "type": f"ALNS Run {run+1}",
            "solution": results['best_solution'],
            "is_feasible": results['best_evaluation']['is_feasible']
        })
    
    # Print summary
    print("\n===== SOLUTION GENERATION DEBUG =====")
    for sol in solutions:
        print(f"{sol['type']} - Feasible: {sol['is_feasible']}")
        print_solution_diagnostics(problem, sol['solution'])
        print("\n")
    
    return solutions

def analyze_problem_constraints(problem):
    """
    Analyze the problem's inherent constraints and potential issues
    
    Args:
        problem: TravelItineraryProblem instance
    """
    print("\n===== PROBLEM CONSTRAINT ANALYSIS =====")
    
    # Basic problem parameters
    print("Problem Parameters:")
    print(f"Number of Days: {problem.NUM_DAYS}")
    print(f"Budget: ${problem.budget:.2f}")
    print(f"Start Time: {problem.START_TIME/60:.1f} hours")
    print(f"End Time: {problem.HARD_LIMIT_END_TIME/60:.1f} hours")
    
    # Locations analysis
    locations = problem.locations
    location_types = {}
    for loc in locations:
        loc_type = loc.get('type', 'unknown')
        location_types[loc_type] = location_types.get(loc_type, 0) + 1
    
    print("\nLocation Types:")
    for loc_type, count in location_types.items():
        print(f"{loc_type.capitalize()}: {count}")
    
    # Meal time windows
    print("\nMeal Time Windows:")
    print(f"Lunch: {problem.LUNCH_START/60:.1f} - {problem.LUNCH_END/60:.1f} hours")
    print(f"Dinner: {problem.DINNER_START/60:.1f} - {problem.DINNER_END/60:.1f} hours")
    
    # Transport matrix analysis
    print("\nTransport Matrix Analysis:")
    transport_matrix = problem.transport_matrix
    route_count = len(transport_matrix)
    time_brackets = set(key[2] for key in transport_matrix.keys())
    
    print(f"Total Routes: {route_count}")
    print(f"Time Brackets: {sorted(time_brackets)}")
    
    # Sample transport times and prices
    print("\nSample Transport Costs:")
    sample_routes = list(transport_matrix.keys())[:5]
    for route in sample_routes:
        print(f"{route}: Transit=${transport_matrix[route]['transit']['price']:.2f}, "
              f"Drive=${transport_matrix[route]['drive']['price']:.2f}")
        
        
def run_diagnostic_checks():
    """
    Comprehensive diagnostic check for travel itinerary problem
    """
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load data
    locations = get_all_locations()
    transport_matrix = get_transport_matrix()
    
    # Create problem instance
    problem = TravelItineraryProblem(
        budget=500,           # Total budget in SGD
        locations=locations,
        transport_matrix=transport_matrix,
        num_days=3            # Trip duration
    )
    
    # Analyze problem constraints
    analyze_problem_constraints(problem)
    
    # Debug solution generation
    debug_solution_generation(problem)

if __name__ == "__main__":
    run_diagnostic_checks()