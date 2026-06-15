# equations_database.py
import re
import math
from typing import Dict, Any, Tuple, Optional, List

EQUATIONS = {
    "Euler Buckling": {
        "formula": "P_{cr} = \\frac{\\pi^2 E I}{L_e^2}",
        "variables": {
            "P_cr": "Critical buckling load (N)",
            "E": "Elastic Modulus (Young's Modulus) (Pa)",
            "I": "Area moment of inertia (m^4)",
            "L_e": "Effective column length (m)"
        },
        "concepts": ["Euler Buckling", "Critical Load", "Slenderness", "Buckling"],
        "solve": lambda vals: (
            ("P_cr", (math.pi**2 * vals["E"] * vals["I"]) / (vals["L_e"]**2)) if "E" in vals and "I" in vals and "L_e" in vals else
            (("L_e", math.sqrt((math.pi**2 * vals["E"] * vals["I"]) / vals["P_cr"])) if "E" in vals and "I" in vals and "P_cr" in vals else
            (("E", (vals["P_cr"] * vals["L_e"]**2) / (math.pi**2 * vals["I"])) if "P_cr" in vals and "L_e" in vals and "I" in vals else
            (("I", (vals["P_cr"] * vals["L_e"]**2) / (math.pi**2 * vals["E"])) if "P_cr" in vals and "L_e" in vals and "E" in vals else None)))
        )
    },
    "Hooke's Law": {
        "formula": "\\sigma = E \\cdot \\epsilon",
        "variables": {
            "\\sigma": "Normal stress (Pa)",
            "E": "Elastic Modulus (Young's Modulus) (Pa)",
            "\\epsilon": "Strain (dimensionless)"
        },
        "concepts": ["Stress", "Strain", "Elastic Deformation", "Yielding"],
        "solve": lambda vals: (
            ("stress", vals["E"] * vals["strain"]) if "E" in vals and "strain" in vals else
            (("strain", vals["stress"] / vals["E"]) if "stress" in vals and "E" in vals else
            (("E", vals["stress"] / vals["strain"]) if "stress" in vals and "strain" in vals else None))
        )
    },
    "Stress": {
        "formula": "\\sigma = \\frac{P}{A}",
        "variables": {
            "\\sigma": "Normal stress (Pa)",
            "P": "Applied axial load (N)",
            "A": "Cross-sectional area (m^2)"
        },
        "concepts": ["Load", "Stress"],
        "solve": lambda vals: (
            ("stress", vals["P"] / vals["A"]) if "P" in vals and "A" in vals else
            (("P", vals["stress"] * vals["A"]) if "stress" in vals and "A" in vals else
            (("A", vals["P"] / vals["stress"]) if "P" in vals and "stress" in vals else None))
        )
    },
    "Newton's Second Law": {
        "formula": "F = m \\cdot a",
        "variables": {
            "F": "Force (N)",
            "m": "Mass (kg)",
            "a": "Acceleration (m/s^2)"
        },
        "concepts": ["Force", "Acceleration", "Newton Second Law"],
        "solve": lambda vals: (
            ("F", vals["m"] * vals["a"]) if "m" in vals and "a" in vals else
            (("m", vals["F"] / vals["a"]) if "F" in vals and "a" in vals else
            (("a", vals["F"] / vals["m"]) if "F" in vals and "m" in vals else None))
        )
    },
    "Torque": {
        "formula": "T = I \\cdot \\alpha",
        "variables": {
            "T": "Torque (N-m)",
            "I": "Moment of inertia (kg-m^2)",
            "\\alpha": "Angular acceleration (rad/s^2)"
        },
        "concepts": ["Torque", "Angular Acceleration", "Moment of Inertia"],
        "solve": lambda vals: (
            ("T", vals["I"] * vals["alpha"]) if "I" in vals and "alpha" in vals else
            (("I", vals["T"] / vals["alpha"]) if "T" in vals and "alpha" in vals else
            (("alpha", vals["T"] / vals["I"]) if "T" in vals and "I" in vals else None))
        )
    },
    "Reynolds Number": {
        "formula": "Re = \\frac{\\rho v D}{\\mu}",
        "variables": {
            "Re": "Reynolds number (dimensionless)",
            "\\rho": "Fluid density (kg/m^3)",
            "v": "Flow speed (m/s)",
            "D": "Characteristic length or pipe diameter (m)",
            "\\mu": "Dynamic viscosity (Pa-s)"
        },
        "concepts": ["Flow Speed", "Reynolds Number", "Turbulence"],
        "solve": lambda vals: (
            ("Re", (vals["density"] * vals["v"] * vals["D"]) / vals["viscosity"]) if "density" in vals and "v" in vals and "D" in vals and "viscosity" in vals else
            (("v", (vals["Re"] * vals["viscosity"]) / (vals["density"] * vals["D"])) if "Re" in vals and "viscosity" in vals and "density" in vals and "D" in vals else None)
        )
    },
    "Fourier's Law": {
        "formula": "q = -k \\frac{dT}{dx}",
        "variables": {
            "q": "Heat flux (W/m^2)",
            "k": "Thermal conductivity (W/m-K)",
            "dT/dx": "Temperature gradient (K/m)"
        },
        "concepts": ["Temperature Gradient", "Fourier Law", "Conduction", "Heat Flux"],
        "solve": lambda vals: (
            ("q", -vals["k"] * vals["gradient"]) if "k" in vals and "gradient" in vals else
            (("gradient", -vals["q"] / vals["k"]) if "q" in vals and "k" in vals else
            (("k", -vals["q"] / vals["gradient"]) if "q" in vals and "gradient" in vals else None))
        )
    },
    "Carnot Efficiency": {
        "formula": "\\eta_C = 1 - \\frac{T_C}{T_H}",
        "variables": {
            "\\eta_C": "Carnot efficiency limit (dimensionless)",
            "T_C": "Cold reservoir temperature (K)",
            "T_H": "Hot reservoir temperature (K)"
        },
        "concepts": ["Carnot cycle", "Efficiency"],
        "solve": lambda vals: (
            ("efficiency", 1.0 - (vals["T_C"] / vals["T_H"])) if "T_C" in vals and "T_H" in vals else
            (("T_C", vals["T_H"] * (1.0 - vals["efficiency"])) if "T_H" in vals and "efficiency" in vals else
            (("T_H", vals["T_C"] / (1.0 - vals["efficiency"])) if "T_C" in vals and "efficiency" in vals else None))
        )
    },
    "Nusselt Number": {
        "formula": "Nu = \\frac{h L}{k}",
        "variables": {
            "Nu": "Nusselt number (dimensionless)",
            "h": "Convective heat transfer coefficient (W/m^2-K)",
            "L": "Characteristic length (m)",
            "k": "Fluid thermal conductivity (W/m-K)"
        },
        "concepts": ["Convection", "Nusselt Number"],
        "solve": lambda vals: (
            ("Nu", (vals["h"] * vals["L"]) / vals["k"]) if "h" in vals and "L" in vals and "k" in vals else
            (("h", (vals["Nu"] * vals["k"]) / vals["L"]) if "Nu" in vals and "k" in vals and "L" in vals else None)
        )
    },
    "Bending Stress": {
        "formula": "\\sigma = \\frac{M y}{I}",
        "variables": {
            "\\sigma": "Bending stress (Pa)",
            "M": "Bending moment (N-m)",
            "y": "Distance from neutral axis (m)",
            "I": "Area moment of inertia (m^4)"
        },
        "concepts": ["Bending Moment", "Euler-Bernoulli Beam", "Bending Stress"],
        "solve": lambda vals: (
            ("stress", (vals["M"] * vals["y"]) / vals["I"]) if "M" in vals and "y" in vals and "I" in vals else
            (("M", (vals["stress"] * vals["I"]) / vals["y"]) if "stress" in vals and "I" in vals and "y" in vals else None)
        )
    }
}

# Regex to extract variables from questions, e.g. E = 200e9, I=1e-5, L_e=3.5, etc.
VAR_PATTERNS = {
    "E": [r"\bE\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "I": [r"\bI\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "L_e": [r"\bL_e\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\bL\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "P_cr": [r"\bP_cr\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\bP\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "stress": [r"\bstress\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\\sigma\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "strain": [r"\bstrain\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\\epsilon\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "P": [r"\bP\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "A": [r"\bA\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "m": [r"\bm\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\bmass\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "a": [r"\ba\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\baccel\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "F": [r"\bF\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\bforce\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "alpha": [r"\balpha\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\bangular_accel\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "T": [r"\bT\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\btorque\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "density": [r"\bdensity\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\\rho\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "v": [r"\bv\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\bspeed\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "D": [r"\bD\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\bdiameter\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "viscosity": [r"\bviscosity\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\\mu\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "Re": [r"\bRe\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "k": [r"\bk\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "gradient": [r"\bgradient\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "q": [r"\bq\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "T_C": [r"\bT_C\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "T_H": [r"\bT_H\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "efficiency": [r"\befficiency\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b", r"\\eta\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "h": [r"\bh\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "Nu": [r"\bNu\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "y": [r"\by\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"],
    "M": [r"\bM\s*=\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"]
}

def extract_variables(text: str) -> Dict[str, float]:
    extracted = {}
    for var, patterns in VAR_PATTERNS.items():
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    extracted[var] = float(match.group(1))
                    break
                except ValueError:
                    pass
    return extracted

def lookup_equations_by_concepts(activated_concepts: List[str]) -> List[Dict[str, Any]]:
    matched = []
    # Make mapping of concepts to lower case
    concepts_lower = [c.lower() for c in activated_concepts]
    for eq_name, data in EQUATIONS.items():
        for eq_concept in data["concepts"]:
            if eq_concept.lower() in concepts_lower:
                matched.append({
                    "name": eq_name,
                    "formula": data["formula"],
                    "variables": data["variables"]
                })
                break
    return matched

def check_and_solve(question_text: str) -> Optional[Dict[str, Any]]:
    # 1. Parse variables from question
    vals = extract_variables(question_text)
    if not vals:
        return None
        
    # 2. Iterate equations to see if we can solve one
    for eq_name, data in EQUATIONS.items():
        result = data["solve"](vals)
        if result:
            solved_var, solved_val = result
            # Format solved value
            if abs(solved_val) >= 1e4 or (abs(solved_val) < 1e-2 and solved_val != 0):
                formatted_val = f"{solved_val:.4e}"
            else:
                formatted_val = f"{solved_val:.4f}"
            return {
                "equation": eq_name,
                "formula": data["formula"],
                "inputs": vals,
                "solved_variable": solved_var,
                "solved_value": formatted_val,
                "variables_desc": data["variables"]
            }
    return None
