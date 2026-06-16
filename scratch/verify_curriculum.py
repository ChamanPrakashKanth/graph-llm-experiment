# scratch/verify_curriculum.py
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equations_database import check_and_solve_chain

def test_solver(name, question, expected_val, expected_var):
    print(f"Testing solver: {name}")
    print(f"Question: '{question}'")
    res = check_and_solve_chain(question, ranked_equations=[name])
    if res is None:
        print(f"FAIL: Solver returned None\n")
        return False
    
    solved_var = res["solved_variable"]
    solved_val = res["solved_value"]
    
    print(f"Solved Variable: {solved_var}")
    print(f"Solved Value: {solved_val} (expected: {expected_val})")
    
    try:
        val_float = float(solved_val)
        expected_float = float(expected_val)
        if abs(val_float - expected_float) < 1e-2:
            print("SUCCESS\n")
            return True
        else:
            print(f"FAIL: Value mismatch ({val_float} vs {expected_float})\n")
            return False
    except ValueError:
        if solved_val == expected_val:
            print("SUCCESS\n")
            return True
        else:
            print(f"FAIL: String value mismatch ('{solved_val}' vs '{expected_val}')\n")
            return False

def main():
    success = True
    
    # 1. Cayley-Hamilton
    success &= test_solver(
        "Cayley-Hamilton Theorem",
        "Given a matrix A with trace of 5.0 and determinant is 6.0, what is the eigenvalue?",
        "3.0000",
        "lambda_val"
    )
    
    # 2. Virtual Work
    success &= test_solver(
        "Virtual Work Principle",
        "If output load is 100 N and output displacement is 0.05 m and input displacement is 0.2 m, find input force",
        "25.0000",
        "F_force"
    )
    
    # 3. Law of Gearing
    success &= test_solver(
        "Law of Gearing",
        "If gear 1 speed is 100 rad/s, pitch radius of gear 1 is 0.05 m, and pitch radius of gear 2 is 0.1 m, find the gear 2 speed",
        "50.0000",
        "omega_2"
    )
    
    # 4. Logarithmic Decrement
    success &= test_solver(
        "Logarithmic Decrement",
        "Determine the logarithmic decrement if damping ratio is 0.1",
        "0.6315",
        "log_dec"
    )
    
    # 5. Soderberg Line
    success &= test_solver(
        "Soderberg Line",
        "Find the factor of safety if mean stress is 100 MPa, yield strength is 300 MPa, stress amplitude is 50 MPa, and endurance limit is 150 MPa",
        "1.5000",
        "FOS"
    )
    
    # 6. Taylor's Tool Life
    success &= test_solver(
        "Taylor's Tool Life Equation",
        "Calculate the tool life in min if cutting speed is 120 m/min, exponent n is 0.25, and constant C is 240",
        "16.0000",
        "tool_life"
    )
    
    # 7. EOQ
    success &= test_solver(
        "Economic Order Quantity (EOQ)",
        "Compute the economic order quantity if annual demand is 10000 units, ordering cost is 50, and holding cost is 4",
        "500.0000",
        "eoq"
    )
    
    if success:
        print("=== ALL SOLVERS VERIFIED SUCCESSFULLY! ===")
    else:
        print("=== SOME SOLVER TESTS FAILED! ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
