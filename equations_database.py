# equations_database.py
import re
import math
from typing import Dict, Any, Tuple, Optional, List

from grammar_parser import normalize_query

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
# Regex to extract variables from questions supporting natural language synonyms and units
VAR_PATTERNS = {
    "E": [
        r"\bE\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\byoung\'s\s+modulus\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\belastic\s+modulus\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "I": [
        r"\bI\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bmoment\s+of\s+inertia\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\barea\s+moment\s+of\s+inertia\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "L_e": [
        r"\bL_e\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\beffective\s+length\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bcolumn\s+length\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bL\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\blength\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "P_cr": [
        r"\bP_cr\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bcritical\s+buckling\s+load\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bbuckling\s+load\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "stress": [
        r"\bstress\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\\sigma\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "strain": [
        r"\bstrain\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\\epsilon\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "P": [
        r"\bload\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bforce\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bP\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "A": [
        r"\barea\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bA\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "m": [
        r"\bmass\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bm\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "a": [
        r"\bacceleration\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\baccel\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\ba\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "F": [
        r"\bforce\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bF\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "alpha": [
        r"\bangular\s+acceleration\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\balpha\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "T": [
        r"\btorque\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bT\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "density": [
        r"\bdensity\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\\rho\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "v": [
        r"\bflowing\s+at\s+([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bspeed\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bvelocity\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bv\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "D": [
        r"\b([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*m\s+diameter\b",
        r"\bdiameter\s*(?:is|=|of)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bpipe\s+diameter\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bD\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "viscosity": [
        r"\bviscosity\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\\mu\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "Re": [
        r"\bRe\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\breynolds\s+number\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "k": [
        r"\bthermal\s+conductivity\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bk\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "gradient": [
        r"\btemperature\s+gradient\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bgradient\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "q": [
        r"\bheat\s+flux\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bq\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "T_C": [
        r"\bcold\s+reservoir\s+temperature\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bT_C\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "T_H": [
        r"\bhot\s+reservoir\s+temperature\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bT_H\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "efficiency": [
        r"\befficiency\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\\eta_C\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\\eta\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "h": [
        r"\bconvective\s+heat\s+transfer\s+coefficient\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bh\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "Nu": [
        r"\bNu\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bnusselt\s+number\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "y": [
        r"\by\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bdistance\s+from\s+neutral\s+axis\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "M": [
        r"\bM\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bbending\s+moment\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ]
}

def parse_value_with_units(val_str: str) -> float:
    # Clean up string
    s = val_str.strip().lower()
    # Find number part and unit part
    match = re.match(r"^([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([a-zA-Z0-9\^/_\-\*]+)?", s)
    if not match:
        raise ValueError(f"Could not parse numeric part from: {val_str}")
    
    num = float(match.group(1))
    unit = match.group(2) if match.group(2) else ""
    
    # Prefix conversion map
    if unit in ("kn", "knewtons"):
        num *= 1e3
    elif unit in ("mn", "mnewtons"):
        num *= 1e6
    elif unit in ("gn", "gnewtons"):
        num *= 1e9
    elif unit in ("mpa", "megapascals"):
        num *= 1e6
    elif unit in ("gpa", "gigapascals"):
        num *= 1e9
    elif unit in ("kpa", "kilopascals"):
        num *= 1e3
    elif unit in ("mm", "millimeters"):
        num *= 1e-3
    elif unit in ("cm", "centimeters"):
        num *= 1e-2
    elif unit in ("mm^2", "mm2"):
        num *= 1e-6
    elif unit in ("cm^2", "cm2"):
        num *= 1e-4
    elif unit in ("mm^4", "mm4"):
        num *= 1e-12
    elif unit in ("cm^4", "cm4"):
        num *= 1e-8
    elif unit in ("kw", "kilowatts"):
        num *= 1e3
    elif unit in ("mw", "megawatts"):
        num *= 1e6
    
    return num

def extract_variables(text: str) -> Dict[str, float]:
    extracted = {}
    for var, patterns in VAR_PATTERNS.items():
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    val_str = match.group(1)
                    extracted[var] = parse_value_with_units(val_str)
                    break
                except Exception as e:
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

# GATE boundary-condition K-factors: L_e = K * L
BOUNDARY_PATTERNS = [
    (r"pinned\s+ends|pinned\s+pinned|both\s+ends\s+pinned", 1.0, "pinned-pinned (K=1.0)"),
    (r"fixed\s+fixed|both\s+ends\s+fixed", 0.5, "fixed-fixed (K=0.5)"),
    (r"fixed\s+pinned|fixed\s+at\s+one\s+end\s+and\s+pinned", 0.7, "fixed-pinned (K=0.7)"),
    (r"fixed\s+free|cantilever|one\s+end\s+fixed.*free", 2.0, "fixed-free / cantilever (K=2.0)"),
]


def _format_value(solved_val: float, output_unit: str = "") -> str:
    if output_unit == "kN" and solved_val >= 1e3:
        solved_val /= 1e3
    elif output_unit == "MPa" and solved_val >= 1e6:
        solved_val /= 1e6
    elif output_unit == "GPa" and solved_val >= 1e9:
        solved_val /= 1e9
    elif output_unit == "kPa" and solved_val >= 1e3:
        solved_val /= 1e3
    if abs(solved_val) >= 1e4 or (abs(solved_val) < 1e-2 and solved_val != 0):
        return f"{solved_val:.4e}"
    return f"{solved_val:.4f}"


def _detect_output_unit(question_text: str, solved_var: str) -> str:
    q = question_text.lower()
    if solved_var in ("P_cr", "P") and "kn" in q:
        return "kN"
    if solved_var == "stress" and "mpa" in q:
        return "MPa"
    return ""


def infer_effective_length(normalized_text: str, vals: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """Infer L_e from physical length L and GATE end-condition phrasing."""
    steps: List[str] = []
    if "L_e" in vals:
        return vals, steps

    L = vals.get("L")
    if L is None:
        m = re.search(
            r"\b(?:column\s+)?length\s+(?:is\s+)?([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)",
            normalized_text, re.IGNORECASE,
        )
        if m:
            L = float(m.group(1))
            vals["L"] = L

    if L is None:
        return vals, steps

    for pattern, k, label in BOUNDARY_PATTERNS:
        if re.search(pattern, normalized_text, re.IGNORECASE):
            vals["L_e"] = k * L
            steps.append(f"End condition: {label} → L_e = {k} × {L} = {vals['L_e']} m")
            break
    return vals, steps


def _solve_equation(eq_name: str, vals: Dict[str, float]) -> Optional[Tuple[str, float]]:
    data = EQUATIONS.get(eq_name)
    if not data:
        return None
    return data["solve"](vals)


def check_and_solve_chain(
    question_text: str,
    concept_path: Optional[List[str]] = None,
    ranked_equations: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Multi-hop symbolic solve: boundary inference → concept-ranked equation chain."""
    normalized = normalize_query(question_text)
    vals = extract_variables(normalized)
    vals, boundary_steps = infer_effective_length(normalized, vals)
    if not vals:
        return None

    eq_order = list(ranked_equations or [])
    for eq_name in EQUATIONS:
        if eq_name not in eq_order:
            eq_order.append(eq_name)

    for eq_name in eq_order:
        result = _solve_equation(eq_name, vals)
        if not result:
            continue
        solved_var, solved_val = result
        out_unit = _detect_output_unit(question_text, solved_var)
        calc_steps = list(boundary_steps)
        calc_steps.append(f"Apply {eq_name}: {EQUATIONS[eq_name]['formula']}")
        formatted = _format_value(solved_val, out_unit)
        if out_unit == "kN":
            calc_steps.append(f"Convert N to kN: {solved_val:.4e} N = {formatted} kN")
        else:
            calc_steps.append(f"Substitute known values → {solved_var} = {formatted}")
        return {
            "equation": eq_name,
            "formula": EQUATIONS[eq_name]["formula"],
            "inputs": vals,
            "solved_variable": solved_var,
            "solved_value": formatted,
            "output_unit": out_unit,
            "variables_desc": EQUATIONS[eq_name]["variables"],
            "calculation_steps": calc_steps,
            "concept_path": concept_path or [],
        }
    return None


def check_and_solve(question_text: str) -> Optional[Dict[str, Any]]:
    return check_and_solve_chain(question_text)
