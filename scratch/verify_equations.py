# scratch/verify_equations.py
import sys
from pathlib import Path

# Fix Windows stdout CP1252 encoding issues with Unicode characters
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure root workspace folder is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from equations_database import check_and_solve_chain

def test_chvorinov():
    print("Testing Chvorinov's Rule:")
    question = "Calculate solidification time if mold constant B is 2.5e6 s/m^2, volume is 0.001 m^3 and surface area is 0.05 m^2."
    result = check_and_solve_chain(question)
    if result:
        print(f" - Solved variable: {result['solved_variable']}")
        print(f" - Solved value: {result['solved_value']}")
        print(f" - Steps: {result['calculation_steps']}")
    else:
        print(" - Failed to solve Chvorinov's Rule")

def test_turbomachinery():
    print("\nTesting Euler Turbomachinery:")
    question = "Compute power if mass flow rate is 10 kg/s, angular velocity is 150 rad/s, inlet radius is 0.2 m, inlet tangential velocity is 5 m/s, outlet radius is 0.3 m, and outlet tangential velocity is 25 m/s."
    result = check_and_solve_chain(question)
    if result:
        print(f" - Solved variable: {result['solved_variable']}")
        print(f" - Solved value: {result['solved_value']}")
        print(f" - Steps: {result['calculation_steps']}")
    else:
        print(" - Failed to solve Euler Turbomachinery")

def test_ofwh():
    print("\nTesting Open Feedwater Heater Energy Balance:")
    question = "Find extraction fraction if h_extracted = 2.8e6, h_feedwater_in = 3.5e5, and exit enthalpy is 7.5e5."
    result = check_and_solve_chain(question)
    if result:
        print(f" - Solved variable: {result['solved_variable']}")
        print(f" - Solved value: {result['solved_value']}")
        print(f" - Steps: {result['calculation_steps']}")
    else:
        print(" - Failed to solve OFWH energy balance")

def test_stability_margins():
    print("\nTesting Stability Margins:")
    question1 = "What is the gain margin if the gain at phase crossover is -12 dB?"
    result1 = check_and_solve_chain(question1)
    if result1:
        print(f" - Solved variable: {result1['solved_variable']}")
        print(f" - Solved value: {result1['solved_value']}")
        print(f" - Steps: {result1['calculation_steps']}")
    else:
        print(" - Failed to solve Gain Margin")
        
    question2 = "Find the phase margin if the phase at gain crossover is -145 degrees."
    result2 = check_and_solve_chain(question2)
    if result2:
        print(f" - Solved variable: {result2['solved_variable']}")
        print(f" - Solved value: {result2['solved_value']}")
        print(f" - Steps: {result2['calculation_steps']}")
    else:
        print(" - Failed to solve Phase Margin")

def main():
    test_chvorinov()
    test_turbomachinery()
    test_ofwh()
    test_stability_margins()

if __name__ == "__main__":
    main()
