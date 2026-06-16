# equations_database.py
import re
import math
from typing import Dict, Any, Tuple, Optional, List

from grammar_parser import normalize_query


def solve_logarithmic_decrement(vals: Dict[str, float]) -> Optional[Tuple[str, float]]:
    delta = None
    if "log_dec" in vals:
        delta = vals["log_dec"]
    elif "damping_ratio" in vals:
        zeta = vals["damping_ratio"]
        if 0 <= zeta < 1.0:
            if vals.get("neglect_higher_powers", 0.0) > 0.5:
                delta = 2 * math.pi * zeta
            else:
                delta = (2 * math.pi * zeta) / math.sqrt(1.0 - zeta**2)

    # 1. Solve for next_peak
    if "initial_peak" in vals and "next_peak" not in vals:
        if delta is not None:
            return "next_peak", vals["initial_peak"] * math.exp(-delta)

    # 2. Solve for initial_peak
    if "next_peak" in vals and "initial_peak" not in vals:
        if delta is not None:
            return "initial_peak", vals["next_peak"] * math.exp(delta)

    # 3. Solve for damping_ratio or log_dec from peaks
    if "initial_peak" in vals and "next_peak" in vals:
        computed_delta = math.log(vals["initial_peak"] / vals["next_peak"])
        if "damping_ratio" not in vals:
            if vals.get("neglect_higher_powers", 0.0) > 0.5:
                zeta = computed_delta / (2.0 * math.pi)
            else:
                zeta = computed_delta / math.sqrt(4.0 * math.pi**2 + computed_delta**2)
            return "damping_ratio", zeta
        if "log_dec" not in vals:
            return "log_dec", computed_delta

    # 4. Standard single step
    if "damping_ratio" in vals and "log_dec" not in vals:
        zeta = vals["damping_ratio"]
        if 0 <= zeta < 1.0:
            if vals.get("neglect_higher_powers", 0.0) > 0.5:
                return "log_dec", 2 * math.pi * zeta
            else:
                return "log_dec", (2 * math.pi * zeta) / math.sqrt(1.0 - zeta**2)

    if "log_dec" in vals and "damping_ratio" not in vals:
        d = vals["log_dec"]
        if d >= 0:
            if vals.get("neglect_higher_powers", 0.0) > 0.5:
                return "damping_ratio", d / (2.0 * math.pi)
            else:
                return "damping_ratio", d / math.sqrt(4.0 * math.pi**2 + d**2)

    return None


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
    },
    "Euler Turbomachinery": {
        "formula": "P = \\dot{m} \\omega (r_2 V_{u2} - r_1 V_{u1})",
        "variables": {
            "P": "Turbomachinery Power (W)",
            "m_dot": "Mass flow rate (kg/s)",
            "omega": "Rotor angular velocity (rad/s)",
            "r_1": "Inlet radius (m)",
            "V_u1": "Inlet absolute tangential velocity (m/s)",
            "r_2": "Outlet radius (m)",
            "V_u2": "Outlet absolute tangential velocity (m/s)"
        },
        "concepts": ["Euler Turbomachinery Equation", "Velocity Triangle", "Turbomachinery Power", "Torque"],
        "solve": lambda vals: (
            ("P", vals["m_dot"] * vals["omega"] * (vals["r_2"] * vals["V_u2"] - vals["r_1"] * vals["V_u1"])) if "m_dot" in vals and "omega" in vals and "r_1" in vals and "V_u1" in vals and "r_2" in vals and "V_u2" in vals else None
        )
    },
    "Open Feedwater Heater Energy Balance": {
        "formula": "y \\cdot h_{extracted} + (1 - y) \\cdot h_{feedwater,in} = h_{out}",
        "variables": {
            "y": "Extraction fraction (dimensionless)",
            "h_extracted": "Extracted steam enthalpy (J/kg)",
            "h_feedwater_in": "Inlet feedwater enthalpy (J/kg)",
            "h_out": "Exit saturated liquid enthalpy (J/kg)"
        },
        "concepts": ["Open Feedwater Heater", "Extraction Fraction", "Energy Balance"],
        "solve": lambda vals: (
            ("y", (vals["h_out"] - vals["h_feedwater_in"]) / (vals["h_extracted"] - vals["h_feedwater_in"])) if "h_out" in vals and "h_feedwater_in" in vals and "h_extracted" in vals and (vals["h_extracted"] - vals["h_feedwater_in"]) != 0 else None
        )
    },
    "Chvorinov's Rule": {
        "formula": "t = B \\left( \\frac{V}{A} \\right)^n",
        "variables": {
            "t": "Solidification time (s)",
            "B_const": "Mold constant (s/m^2)",
            "volume": "Volume of casting (m^3)",
            "A": "Surface area of casting (m^2)",
            "n_exp": "Exponent (usually 2)"
        },
        "concepts": ["Chvorinov's Rule", "Solidification Time", "Casting Modulus", "Mold Constant", "Riser Design"],
        "solve": lambda vals: (
            ("t", vals["B_const"] * (vals["volume"] / vals["A"])**vals.get("n_exp", 2.0)) if "B_const" in vals and "volume" in vals and "A" in vals and vals["A"] != 0 else None
        )
    },
    "Gain Margin": {
        "formula": "GM_{dB} = 0 - |G(j\\omega_{pc})|_{dB}",
        "variables": {
            "GM_dB": "Gain margin (dB)",
            "gain_at_pc": "Magnitude at phase crossover frequency (dB)"
        },
        "concepts": ["Bode Plot", "Phase Crossover Frequency", "Gain Margin", "Closed Loop Stability"],
        "solve": lambda vals: (
            ("GM_dB", 0.0 - vals["gain_at_pc"]) if "gain_at_pc" in vals else None
        )
    },
    "Phase Margin": {
        "formula": "PM = 180^\\circ + \\angle G(j\\omega_{gc})",
        "variables": {
            "PM": "Phase margin (degrees)",
            "phase_at_gc": "Phase angle at gain crossover frequency (degrees)"
        },
        "concepts": ["Bode Plot", "Gain Crossover Frequency", "Phase Margin", "Closed Loop Stability"],
        "solve": lambda vals: (
            ("PM", 180.0 + vals["phase_at_gc"]) if "phase_at_gc" in vals else None
        )
    },
    "Cayley-Hamilton Theorem": {
        "formula": "\\lambda^2 - tr(A) \\cdot \\lambda + det(A) = 0",
        "variables": {
            "lambda_val": "Eigenvalue (dimensionless)",
            "tr_A": "Trace of matrix A (dimensionless)",
            "det_A": "Determinant of matrix A (dimensionless)"
        },
        "concepts": ["Cayley-Hamilton Theorem", "Eigenvalue", "Matrix", "Characteristic Polynomial"],
        "solve": lambda vals: (
            ("lambda_val", (vals["tr_A"] + math.sqrt(vals["tr_A"]**2 - 4 * vals["det_A"])) / 2.0) if "tr_A" in vals and "det_A" in vals and (vals["tr_A"]**2 - 4 * vals["det_A"]) >= 0 else (
            ("det_A", vals["tr_A"] * vals["lambda_val"] - vals["lambda_val"]**2) if "tr_A" in vals and "lambda_val" in vals else (
            ("tr_A", (vals["lambda_val"]**2 + vals["det_A"]) / vals["lambda_val"]) if "lambda_val" in vals and "det_A" in vals and vals["lambda_val"] != 0 else None
            )
            )
        )
    },
    "Virtual Work Principle": {
        "formula": "F_{force} \\cdot \\delta x = P_{load} \\cdot \\delta y",
        "variables": {
            "F_force": "Input force (N)",
            "delta_x": "Virtual displacement of input (m)",
            "P_load": "Output load force (N)",
            "delta_y": "Virtual displacement of output (m)"
        },
        "concepts": ["Virtual Work Principle", "Static Equilibrium", "Virtual Displacement"],
        "solve": lambda vals: (
            ("F_force", (vals["P_load"] * vals["delta_y"]) / vals["delta_x"]) if "P_load" in vals and "delta_y" in vals and "delta_x" in vals and vals["delta_x"] != 0 else (
            ("P_load", (vals["F_force"] * vals["delta_x"]) / vals["delta_y"]) if "F_force" in vals and "delta_x" in vals and "delta_y" in vals and vals["delta_y"] != 0 else (
            ("delta_x", (vals["P_load"] * vals["delta_y"]) / vals["F_force"]) if "P_load" in vals and "delta_y" in vals and "F_force" in vals and vals["F_force"] != 0 else (
            ("delta_y", (vals["F_force"] * vals["delta_x"]) / vals["P_load"]) if "F_force" in vals and "delta_x" in vals and "P_load" in vals and vals["P_load"] != 0 else None
            )
            )
            )
        )
    },
    "Law of Gearing": {
        "formula": "\\omega_1 \\cdot R_1 = \\omega_2 \\cdot R_2",
        "variables": {
            "omega_1": "Angular velocity of gear 1 (rad/s)",
            "R_1": "Pitch radius of gear 1 (m)",
            "omega_2": "Angular velocity of gear 2 (rad/s)",
            "R_2": "Pitch radius of gear 2 (m)"
        },
        "concepts": ["Law of Gearing", "Angular Velocity", "Velocity Ratio", "Gear"],
        "solve": lambda vals: (
            ("omega_1", (vals["omega_2"] * vals["R_2"]) / vals["R_1"]) if "omega_2" in vals and "R_2" in vals and "R_1" in vals and vals["R_1"] != 0 else (
            ("omega_2", (vals["omega_1"] * vals["R_1"]) / vals["R_2"]) if "omega_1" in vals and "R_1" in vals and "R_2" in vals and vals["R_2"] != 0 else (
            ("R_1", (vals["omega_2"] * vals["R_2"]) / vals["omega_1"]) if "omega_2" in vals and "R_2" in vals and "omega_1" in vals and vals["omega_1"] != 0 else (
            ("R_2", (vals["omega_1"] * vals["R_1"]) / vals["omega_2"]) if "omega_1" in vals and "R_1" in vals and "omega_2" in vals and vals["omega_2"] != 0 else None
            )
            )
            )
        )
    },
    "Logarithmic Decrement": {
        "formula": "\\delta = \\frac{2 \\pi \\zeta}{\\sqrt{1 - \\zeta^2}}",
        "variables": {
            "log_dec": "Logarithmic decrement (dimensionless)",
            "damping_ratio": "Damping ratio (dimensionless)",
            "initial_peak": "Initial displacement peak (m)",
            "next_peak": "Displacement peak at next cycle (m)"
        },
        "concepts": ["Logarithmic Decrement", "Damping Ratio", "Damped Vibration"],
        "solve": solve_logarithmic_decrement
    },
    "Soderberg Line": {
        "formula": "\\frac{\\sigma_a}{S_e} + \\frac{\\sigma_m}{S_y} = \\frac{1}{FOS}",
        "variables": {
            "sigma_a": "Stress amplitude (Pa)",
            "S_e": "Endurance limit (Pa)",
            "sigma_m": "Mean stress (Pa)",
            "S_y": "Yield strength (Pa)",
            "FOS": "Factor of Safety (dimensionless)"
        },
        "concepts": ["Soderberg Line", "Fatigue Design", "Stress Amplitude", "Mean Stress", "Yield Strength", "Endurance Limit"],
        "solve": lambda vals: (
            ("FOS", 1.0 / (vals["sigma_a"] / vals["S_e"] + vals["sigma_m"] / vals["S_y"])) if "sigma_a" in vals and "S_e" in vals and "sigma_m" in vals and "S_y" in vals and (vals["sigma_a"] / vals["S_e"] + vals["sigma_m"] / vals["S_y"]) != 0 else (
            ("sigma_a", (1.0 / vals["FOS"] - vals["sigma_m"] / vals["S_y"]) * vals["S_e"]) if "FOS" in vals and "sigma_m" in vals and "S_y" in vals and "S_e" in vals else (
            ("sigma_m", (1.0 / vals["FOS"] - vals["sigma_a"] / vals["S_e"]) * vals["S_y"]) if "FOS" in vals and "sigma_a" in vals and "S_e" in vals and "S_y" in vals else (
            ("S_e", vals["sigma_a"] / (1.0 / vals["FOS"] - vals["sigma_m"] / vals["S_y"])) if "FOS" in vals and "sigma_m" in vals and "S_y" in vals and "sigma_a" in vals and (1.0 / vals["FOS"] - vals["sigma_m"] / vals["S_y"]) != 0 else (
            ("S_y", vals["sigma_m"] / (1.0 / vals["FOS"] - vals["sigma_a"] / vals["S_e"])) if "FOS" in vals and "sigma_a" in vals and "S_e" in vals and "sigma_m" in vals and (1.0 / vals["FOS"] - vals["sigma_a"] / vals["S_e"]) != 0 else None
            )
            )
            )
            )
        )
    },
    "Taylor's Tool Life Equation": {
        "formula": "V \\cdot T^n = C",
        "variables": {
            "cutting_speed": "Cutting speed (m/min)",
            "tool_life": "Tool life (min)",
            "taylor_n": "Exponent n (dimensionless)",
            "taylor_c": "Constant C (dimensionless)"
        },
        "concepts": ["Taylor's Tool Life Equation", "Tool Wear", "Cutting Speed"],
        "solve": lambda vals: (
            ("taylor_c", vals["cutting_speed"] * (vals["tool_life"] ** vals["taylor_n"])) if "cutting_speed" in vals and "tool_life" in vals and "taylor_n" in vals else (
            ("cutting_speed", vals["taylor_c"] / (vals["tool_life"] ** vals["taylor_n"])) if "taylor_c" in vals and "tool_life" in vals and "taylor_n" in vals and vals["tool_life"] != 0 else (
            ("tool_life", (vals["taylor_c"] / vals["cutting_speed"]) ** (1.0 / vals["taylor_n"])) if "taylor_c" in vals and "cutting_speed" in vals and "taylor_n" in vals and vals["cutting_speed"] != 0 and vals["taylor_n"] != 0 else (
            ("taylor_n", math.log(vals["taylor_c"] / vals["cutting_speed"]) / math.log(vals["tool_life"])) if "taylor_c" in vals and "cutting_speed" in vals and "tool_life" in vals and vals["cutting_speed"] != 0 and vals["tool_life"] > 0 and vals["tool_life"] != 1.0 else None
            )
            )
            )
        )
    },
    "Economic Order Quantity (EOQ)": {
        "formula": "Q = \\sqrt{\\frac{2 D S}{H}}",
        "variables": {
            "annual_demand": "Annual demand (units/year)",
            "ordering_cost": "Ordering cost (currency/order)",
            "holding_cost": "Holding cost (currency/unit-year)",
            "eoq": "Economic Order Quantity (units)"
        },
        "concepts": ["Economic Order Quantity (EOQ)", "Inventory Cost", "Holding Cost", "Ordering Cost"],
        "solve": lambda vals: (
            ("eoq", math.sqrt((2.0 * vals["annual_demand"] * vals["ordering_cost"]) / vals["holding_cost"])) if "annual_demand" in vals and "ordering_cost" in vals and "holding_cost" in vals and vals["holding_cost"] > 0 else (
            ("annual_demand", (vals["eoq"]**2 * vals["holding_cost"]) / (2.0 * vals["ordering_cost"])) if "eoq" in vals and "holding_cost" in vals and "ordering_cost" in vals and vals["ordering_cost"] > 0 else (
            ("ordering_cost", (vals["eoq"]**2 * vals["holding_cost"]) / (2.0 * vals["annual_demand"])) if "eoq" in vals and "holding_cost" in vals and "annual_demand" in vals and vals["annual_demand"] > 0 else (
            ("holding_cost", (2.0 * vals["annual_demand"] * vals["ordering_cost"]) / vals["eoq"]**2) if "annual_demand" in vals and "ordering_cost" in vals and "eoq" in vals and vals["eoq"] > 0 else None
            )
            )
            )
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
    ],
    "m_dot": [
        r"\bm_dot\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bmass\s+flow\s+rate\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "omega": [
        r"\bomega\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bangular\s+velocity\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\brotational\s+speed\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "r_1": [
        r"\br_1\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\binlet\s+radius\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "V_u1": [
        r"\bV_u1\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\binlet\s+tangential\s+velocity\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "r_2": [
        r"\br_2\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\boutlet\s+radius\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "V_u2": [
        r"\bV_u2\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\boutlet\s+tangential\s+velocity\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "h_extracted": [
        r"\bh_extracted\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bextracted\s+steam\s+enthalpy\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bsteam\s+enthalpy\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "h_feedwater_in": [
        r"\bh_feedwater_in\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\binlet\s+feedwater\s+enthalpy\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "h_out": [
        r"\bh_out\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bexit\s+feedwater\s+enthalpy\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bexit\s+enthalpy\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "volume": [
        r"\bvolume\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bV\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "B_const": [
        r"\bB\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bmold\s+constant\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "n_exp": [
        r"\bn\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bexponent\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "gain_at_pc": [
        r"\bgain\s+at\s+phase\s+crossover\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bmagnitude\s+at\s+phase\s+crossover\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bmagnitude\s+at\s+w_pc\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "phase_at_gc": [
        r"\bphase\s+at\s+gain\s+crossover\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bphase\s+angle\s+at\s+gain\s+crossover\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bphase\s+at\s+w_gc\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "tr_A": [
        r"\btrace\s*(?:is|=|of)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\btr\(A\)\s*(?:is|=|of)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "det_A": [
        r"\bdeterminant\s*(?:is|=|of)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bdet\(A\)\s*(?:is|=|of)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "lambda_val": [
        r"\beigenvalue\s*(?:is|=|of)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\blambda\s*(?:is|=|of)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "F_force": [
        r"\binput\s+force\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bF_force\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "delta_x": [
        r"\binput\s+displacement\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bdelta\s+x\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "P_load": [
        r"\boutput\s+load\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bload\s+force\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bP_load\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "delta_y": [
        r"\boutput\s+displacement\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bdelta\s+y\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "omega_1": [
        r"\bomega_1\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bgear\s+1\s+speed\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bangular\s+velocity\s+of\s+gear\s+1\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "R_1": [
        r"\bR_1\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bradius\s+of\s+gear\s+1\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bpitch\s+radius\s+of\s+gear\s+1\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "omega_2": [
        r"\bomega_2\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bgear\s+2\s+speed\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bangular\s+velocity\s+of\s+gear\s+2\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "R_2": [
        r"\bR_2\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bradius\s+of\s+gear\s+2\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bpitch\s+radius\s+of\s+gear\s+2\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "log_dec": [
        r"\blogarithmic\s+decrement\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\blog\s+decrement\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bdelta\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "damping_ratio": [
        r"\bdamping\s+ratio\b.*?\b([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\\zeta\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "initial_peak": [
        r"\bdisplacement\s+peak\b.*?\b([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\binitial\s+peak\b.*?\b([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bpeak\s+displacement\b.*?\b([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bx_0\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bx0\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "next_peak": [
        r"\bnext\s+peak\b.*?\b([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bx_1\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bx1\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "sigma_a": [
        r"\bstress\s+amplitude\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bsigma_a\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bamplitude\s+stress\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "S_e": [
        r"\bendurance\s+limit\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bendurance\s+strength\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bS_e\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "sigma_m": [
        r"\bmean\s+stress\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bsigma_m\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "S_y": [
        r"\byield\s+strength\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\byield\s+point\s+stress\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bS_y\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "FOS": [
        r"\bfactor\s+of\s+safety\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bfos\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "cutting_speed": [
        r"\bcutting\s+speed\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bcutting\s+velocity\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bV\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "tool_life": [
        r"\btool\s+life\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b",
        r"\bT\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?(?:\s*[a-zA-Z0-9\^/_\-\*]+)?)\b"
    ],
    "taylor_n": [
        r"\btaylor\s+exponent\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\btool\s+life\s+exponent\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bexponent\s+n\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bn\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "taylor_c": [
        r"\btaylor\s+constant\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\btool\s+life\s+constant\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bconstant\s+C\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bC\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "annual_demand": [
        r"\bannual\s+demand\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bdemand\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bD\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "ordering_cost": [
        r"\bordering\s+cost\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bsetup\s+cost\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bS\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "holding_cost": [
        r"\bholding\s+cost\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bcarrying\s+cost\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bH\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
    ],
    "eoq": [
        r"\beoq\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\beconomic\s+order\s+quantity\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b",
        r"\bQ\s*(?:is|=)\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\b"
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
    elif output_unit == "mm":
        solved_val /= 1e-3
    if abs(solved_val) >= 1e4 or (abs(solved_val) < 1e-2 and solved_val != 0):
        return f"{solved_val:.4e}"
    return f"{solved_val:.4f}"


def _detect_output_unit(question_text: str, solved_var: str) -> str:
    q = question_text.lower()
    if solved_var in ("P_cr", "P", "F_force", "P_load") and "kn" in q:
        return "kN"
    if solved_var in ("stress", "sigma_a", "sigma_m", "S_e", "S_y") and "mpa" in q:
        return "MPa"
    if solved_var in ("initial_peak", "next_peak", "L_e", "L", "D", "r_1", "r_2") and "mm" in q:
        return "mm"
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
    if "neglect" in normalized.lower():
        vals["neglect_higher_powers"] = 1.0
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
