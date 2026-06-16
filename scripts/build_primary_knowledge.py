# scripts/build_primary_knowledge.py
import json
import os
import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure root folder is in python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from reasoning_graph import ReasoningGraph

SOURCE_NAME = "Primary Engineering Textbooks (GATE ME Curriculum)"

CONCEPTS_DATA = [
    # 1. Thermodynamics
    {
        "concept": "First Law",
        "definition": "The law of conservation of energy stating that energy cannot be created or destroyed, only transformed from one form to another.",
        "formula": "dQ = dU + dW",
        "causes": ["Heat transfer", "Work interaction"],
        "effects": ["Change in internal energy", "Temperature change", "Boundary work"],
        "dependencies": ["Heat interaction", "Work interaction", "State variables"],
        "applications": ["Internal combustion engines", "Steam power plants", "Gas turbines"],
        "failure_modes": ["Energy leakage", "Thermal insulation failure"],
        "related_concepts": ["Thermodynamics", "Energy", "Entropy"],
        "prerequisites": ["Thermodynamics", "Energy"],
        "difficulty": "Easy",
        "revision_1_line": "Conservation of energy: dQ = dU + dW.",
        "revision_5_lines": "Energy is conserved in all processes.\nHeat added equals change in internal energy plus work done.\nForms the basis of cycle energy analysis.\nValid for both closed and open systems.\nExcludes quality of energy considerations.",
        "explanation": "The First Law of Thermodynamics is the thermodynamic formulation of the conservation of energy principle. It relates heat added to a system, work done by the system, and changes in the internal energy of the system.",
        "exam_summary": "Core basis for almost all thermal system problems in GATE. Remember the sign convention: heat in is positive, work out is positive.",
        "typical_mistakes": "Incorrect sign convention for work or heat, or neglecting changes in kinetic and potential energy in steady flow systems.",
        "memory_aid": "Energy in = Energy out + Accumulation.",
        "edges": [
            ("First Law", "Thermodynamics", "part_of"),
            ("First Law", "Power Cycles", "prerequisite_for"),
            ("First Law", "Entropy", "depends_on")
        ]
    },
    {
        "concept": "Second Law",
        "definition": "A law stating that heat cannot spontaneously flow from a cooler body to a warmer body, and the entropy of an isolated system always increases.",
        "formula": "eta_max = 1 - T_C / T_H",
        "causes": ["Irreversibilities", "Temperature differences"],
        "effects": ["Entropy generation", "Degradation of energy quality"],
        "dependencies": ["Thermal reservoirs", "Temperature gradients"],
        "applications": ["Heat engines", "Refrigerators", "Heat pumps"],
        "failure_modes": ["Excessive irreversibility", "Thermal degradation"],
        "related_concepts": ["Carnot cycle", "Entropy", "Thermal Efficiency"],
        "prerequisites": ["Thermodynamics", "First Law"],
        "difficulty": "Medium",
        "revision_1_line": "Heat cannot spontaneously flow from cold to hot.",
        "revision_5_lines": "Asserts that real processes are directional.\nDefines the limit of thermal efficiency.\nIntroduces the concept of entropy.\nClausius and Kelvin-Planck statements are equivalent.\nDetermines if a cycle is reversible or irreversible.",
        "explanation": "The Second Law of Thermodynamics dictates that energy has quality as well as quantity. It limits the efficiency of cycles and establishes the direction of spontaneous heat transfer.",
        "exam_summary": "Highly tested in GATE. Used to check proposed engines for feasibility using Clausius Inequality or Carnot limits.",
        "typical_mistakes": "Forgetting that work can easily convert to heat, but heat cannot convert 100% to work.",
        "memory_aid": "Heat flows downhill naturally.",
        "edges": [
            ("Second Law", "Thermodynamics", "part_of"),
            ("Second Law", "Entropy", "leads_to"),
            ("Second Law", "Carnot cycle", "governs")
        ]
    },
    {
        "concept": "Entropy",
        "definition": "A thermodynamic property that measures the degree of molecular disorder or randomness in a system.",
        "formula": "dS = dQ_rev / T",
        "causes": ["Heat transfer", "Friction", "Unrestrained expansion"],
        "effects": ["Increase in disorder", "Decrease in available energy"],
        "dependencies": ["Temperature", "Heat transfer", "System irreversibility"],
        "applications": ["Thermal efficiency calculation", "Cycle analysis", "Exergy destruction"],
        "failure_modes": ["Exergy destruction", "Entropy generation"],
        "related_concepts": ["Second Law", "Clausius Inequality", "Exergy"],
        "prerequisites": ["Thermodynamics", "Second Law"],
        "difficulty": "Hard",
        "revision_1_line": "Measure of molecular disorder and process irreversibility.",
        "revision_5_lines": "Entropy is a state function.\nEntropy of an isolated system always increases or stays constant.\nGenerates during any real (irreversible) process.\ndS >= dQ/T holds for all processes.\nZero entropy exists for a perfect crystal at absolute zero.",
        "explanation": "Entropy represents the dispersal of energy. It is used to quantify the irreversibility of thermal systems and determine how close a system is to equilibrium.",
        "exam_summary": "GATE questions focus on calculating entropy change for ideal gases and reservoirs: dS = m*c_p*ln(T2/T1) - m*R*ln(P2/P1).",
        "typical_mistakes": "Using Celsius instead of Kelvin for temperature, or neglecting entropy change of the surroundings.",
        "memory_aid": "Entropy is like room clutter; it increases unless work is done.",
        "edges": [
            ("Entropy", "Second Law", "derived_from"),
            ("Entropy", "Clausius Inequality", "depends_on"),
            ("Entropy", "Thermodynamics", "part_of")
        ]
    },
    {
        "concept": "Availability",
        "definition": "The maximum useful work obtainable from a system as it comes into equilibrium with the environment.",
        "formula": "Phi = (U - U_0) + P_0*(V - V_0) - T_0*(S - S_0)",
        "causes": ["Exergy input", "Temperature difference from surroundings"],
        "effects": ["Potential for work output", "Power generation"],
        "dependencies": ["Environment temperature", "System state", "Environment pressure"],
        "applications": ["Exergy analysis", "Power plant optimization", "Thermal efficiency improvement"],
        "failure_modes": ["Exergy destruction", "Irreversibility losses"],
        "related_concepts": ["Second Law", "Entropy", "Useful Work"],
        "prerequisites": ["Entropy", "First Law"],
        "difficulty": "Hard",
        "revision_1_line": "Maximum possible work output from a system relative to environment.",
        "revision_5_lines": "Also known as exergy.\nRepresents the quality component of energy.\nExergy is destroyed by irreversibilities.\nMaximum work is obtained via a reversible process.\nDead state is when system is in equilibrium with surroundings.",
        "explanation": "Availability (exergy) analysis evaluates thermodynamic performance by identifying where work potential is lost (exergy destruction), allowing engineers to optimize plant designs.",
        "exam_summary": "Tested in exergy analysis of open/closed systems in GATE. Calculate change in exergy: dPhi = dH - T_0*dS.",
        "typical_mistakes": "Confusing exergy with energy, or failing to use absolute temperature for surroundings T_0.",
        "memory_aid": "Availability = energy that can actually do work.",
        "edges": [
            ("Availability", "Entropy", "depends_on"),
            ("Availability", "Second Law", "requires"),
            ("Availability", "Thermodynamics", "part_of")
        ]
    },
    {
        "concept": "Power Cycles",
        "definition": "Thermodynamic cycles designed to continuously convert heat input into net work output.",
        "formula": "eta_thermal = W_net / Q_in",
        "causes": ["Thermal energy input", "Combustion", "Nuclear fission"],
        "effects": ["Electricity generation", "Mechanical power"],
        "dependencies": ["Working fluid", "Turbine", "Condenser", "Combustor"],
        "applications": ["Rankine cycle plants", "Otto cycle engines", "Diesel engines"],
        "failure_modes": ["Turbine blade failure", "Condenser leakage", "Heat loss"],
        "related_concepts": ["Thermal Efficiency", "Clausius Inequality", "Rankine Cycle"],
        "prerequisites": ["Thermodynamics", "First Law"],
        "difficulty": "Medium",
        "revision_1_line": "Cycles converting heat into useful mechanical work.",
        "revision_5_lines": "Rankine, Otto, Diesel, and Brayton are key power cycles.\nEfficiency is limited by Carnot efficiency.\nRequires a high-temp source and low-temp sink.\nWorking fluid can undergo phase change or remain gaseous.\nOptimized by reheating, regeneration, or intercooling.",
        "explanation": "Power cycles model the operational processes of heat engines. Standard gas power cycles assume air-standard assumptions, while vapor power cycles model steam behaviour in power plants.",
        "exam_summary": "Highly tested. Focus on cycle diagrams (P-V, T-s) and efficiency formulas for Rankine, Otto, and Brayton cycles.",
        "typical_mistakes": "Confusing Brayton cycle (gas) with Rankine cycle (vapor) equations, or misidentifying cycle processes.",
        "memory_aid": "Heat in, Work out, Waste heat rejected.",
        "edges": [
            ("Power Cycles", "Thermodynamics", "part_of"),
            ("Power Cycles", "First Law", "requires"),
            ("Power Cycles", "Clausius Inequality", "affects")
        ]
    },
    {
        "concept": "Refrigeration",
        "definition": "The process of removing heat from a low-temperature space and transferring it to a higher-temperature environment.",
        "formula": "COP = Q_L / W_in",
        "causes": ["Compressor work input", "Refrigerant expansion"],
        "effects": ["Temperature drop in refrigerated space"],
        "dependencies": ["Refrigerant", "Compressor", "Condenser", "Evaporator"],
        "applications": ["Air conditioning", "Food preservation", "Gas liquefaction"],
        "failure_modes": ["Refrigerant leakage", "Compressor seizure", "Expansion valve block"],
        "related_concepts": ["Coefficient of Performance (COP)", "Heat Pump", "Carnot cycle"],
        "prerequisites": ["Thermodynamics", "Second Law"],
        "difficulty": "Medium",
        "revision_1_line": "Moving heat from cold to hot using work input.",
        "revision_5_lines": "Opposite of a heat engine cycle.\nPerformance measured by Coefficient of Performance (COP).\nCOP can be greater than 1.0.\nVapor compression is the most common cycle.\nUses throttling expansion rather than turbine expansion.",
        "explanation": "Refrigeration cycles absorb heat at low temperatures through an evaporator, compress the refrigerant, release heat at high temperatures in a condenser, and expand the fluid via a throttling valve.",
        "exam_summary": "GATE tests Vapor Compression Refrigeration Cycle (VCRC). Calculate COP = (h1 - h4) / (h2 - h1) where h is enthalpy.",
        "typical_mistakes": "Swapping evaporator and condenser enthalpy values, or using turbine work instead of throttling.",
        "memory_aid": "COP is desired effect divided by work input.",
        "edges": [
            ("Refrigeration", "Thermodynamics", "part_of"),
            ("Refrigeration", "Second Law", "requires"),
            ("Refrigeration", "Carnot cycle", "depends_on")
        ]
    },
    {
        "concept": "Gas Turbines",
        "definition": "Internal combustion engines that convert heat from fuel combustion into mechanical energy using rotating turbine blades.",
        "formula": "W_net = W_t - W_c",
        "causes": ["High-pressure gas expansion", "Combustion"],
        "effects": ["Thrust generation", "Electrical power"],
        "dependencies": ["Compressor", "Combustor", "Turbine", "Brayton cycle"],
        "applications": ["Jet engines", "Power generation", "Marine propulsion"],
        "failure_modes": ["High-temperature blade creep", "Compressor stall", "Thermal fatigue"],
        "related_concepts": ["Brayton Cycle", "Turbine", "Thermal Stress"],
        "prerequisites": ["Power Cycles", "Thermodynamics"],
        "difficulty": "Medium",
        "revision_1_line": "Brayton-cycle rotary engines for thrust and power.",
        "revision_5_lines": "Operates on the gas Brayton cycle.\nHigh back-work ratio compared to steam plants.\nOpen cycle in practice, closed cycle in theory.\nCombustion occurs at constant pressure.\nEfficiency increases with pressure ratio.",
        "explanation": "Gas turbines pull air into a compressor, heat it in a combustion chamber, and expand it through a turbine. The turbine drives both the compressor and the load (e.g. generator or propeller).",
        "exam_summary": "Common GATE questions cover Brayton cycle efficiency: eta = 1 - 1 / (r_p)^((gamma-1)/gamma).",
        "typical_mistakes": "Forgetting that compressor work is a huge fraction of turbine work, unlike Rankine cycle pumps.",
        "memory_aid": "Suck, Squeeze, Bang, Blow.",
        "edges": [
            ("Gas Turbines", "Thermodynamics", "part_of"),
            ("Gas Turbines", "Power Cycles", "used_in"),
            ("Gas Turbines", "Turbine", "depends_on")
        ]
    },

    # 2. Fluid Mechanics
    {
        "concept": "Bernoulli Equation",
        "definition": "A relation between pressure, velocity, and elevation in steady, incompressible, frictionless flow.",
        "formula": "P + 0.5 * rho * v^2 + rho * g * z = Constant",
        "causes": ["Gravity", "Pressure forces", "Flow velocity changes"],
        "effects": ["Lift generation", "Pressure drop at high speed"],
        "dependencies": ["Incompressible flow", "Steady flow", "Inviscid flow"],
        "applications": ["Venturimeter", "Pitot tube", "Orifice meter"],
        "failure_modes": ["Flow separation", "Cavitation"],
        "related_concepts": ["Fluid Mechanics", "Flow Speed", "Venturimeter"],
        "prerequisites": ["Fluid Mechanics", "Newton Second Law"],
        "difficulty": "Medium",
        "revision_1_line": "Energy conservation along a streamline for inviscid fluid flow.",
        "revision_5_lines": "Valid only along a streamline for rotational flow.\nValid anywhere in the flow field for irrotational flow.\nAssumes frictionless, inviscid fluid.\nAssumes constant density (incompressible).\nSum of pressure head, velocity head, and datum head is constant.",
        "explanation": "The Bernoulli Equation is an approximate relation between pressure, velocity, and elevation. It is a statement of conservation of mechanical energy for a fluid particle along a streamline.",
        "exam_summary": "Highly tested. Used to solve flow rate in venturimeters: Q = C_d * A1 * A2 * sqrt(2*g*h) / sqrt(A1^2 - A2^2).",
        "typical_mistakes": "Applying it to highly viscous flows, compressible gases, or unsteady flow without proper terms.",
        "memory_aid": "Higher velocity means lower pressure.",
        "edges": [
            ("Bernoulli Equation", "Fluid Mechanics", "part_of"),
            ("Bernoulli Equation", "Venturimeter", "governs"),
            ("Bernoulli Equation", "Flow Speed", "controls")
        ]
    },
    {
        "concept": "Boundary Layer",
        "definition": "The thin layer of fluid adjacent to a solid surface where viscous forces are significant.",
        "formula": "delta = 5.0 * x / sqrt(Re_x)",
        "causes": ["No-slip boundary condition", "Viscous drag"],
        "effects": ["Skin friction drag", "Velocity gradient development"],
        "dependencies": ["Fluid viscosity", "Surface roughness", "Flow velocity"],
        "applications": ["Aerodynamic drag reduction", "Heat transfer coefficient estimation", "Aircraft design"],
        "failure_modes": ["Boundary layer separation", "Stall"],
        "related_concepts": ["Viscous Flow", "Skin Friction Drag", "Navier-Stokes Equations"],
        "prerequisites": ["Fluid Mechanics", "Reynolds Number"],
        "difficulty": "Hard",
        "revision_1_line": "Thin fluid layer near surface dominated by viscosity.",
        "revision_5_lines": "Velocity goes from zero at surface to free-stream speed.\nBoundary layer thickens downstream.\nTransitions from laminar to turbulent at critical Re.\nGoverned by Prandtl boundary layer equations.\nSeparates when facing an adverse pressure gradient.",
        "explanation": "Because of viscosity, fluid particles in contact with a solid surface come to a complete stop (no-slip condition). This creates a velocity gradient and shear stress in the boundary layer near the wall.",
        "exam_summary": "Tested in GATE. Know thickness formulas for laminar: delta proportional to x^0.5, and turbulent: delta proportional to x^0.8.",
        "typical_mistakes": "Confusing displacement thickness, momentum thickness, and boundary layer thickness definitions.",
        "memory_aid": "Boundary layer is where fluid feels the wall friction.",
        "edges": [
            ("Boundary Layer", "Fluid Mechanics", "part_of"),
            ("Boundary Layer", "Viscous Flow", "depends_on"),
            ("Boundary Layer", "Skin Friction Drag", "causes")
        ]
    },
    {
        "concept": "Pipe Flow",
        "definition": "Flow of fluid inside closed conduits, driven by pressure gradients.",
        "formula": "h_f = f * L * v^2 / (2 * g * D)",
        "causes": ["Pressure drop", "Pumping power"],
        "effects": ["Fluid transportation", "Friction losses"],
        "dependencies": ["Pipe diameter", "Roughness", "Pipe length", "Flow speed"],
        "applications": ["Water distribution networks", "Oil pipelines", "Piping design"],
        "failure_modes": ["Pipe bursting", "Scaling", "Excessive pressure drop"],
        "related_concepts": ["Reynolds Number", "Darcy-Weisbach Equation", "Friction Factor"],
        "prerequisites": ["Fluid Mechanics", "Reynolds Number"],
        "difficulty": "Medium",
        "revision_1_line": "Viscous flow inside conduits governed by friction factor.",
        "revision_5_lines": "Laminar flow occurs when Re < 2000.\nTurbulent flow occurs when Re > 4000.\nFriction factor f = 64/Re for laminar flow.\nFriction factor depends on roughness for turbulent flow.\nHead loss is calculated using Darcy-Weisbach equation.",
        "explanation": "Pipe flow is a common fluid application. Friction along the pipe walls causes a pressure drop (head loss), which must be compensated by pumps to maintain flow.",
        "exam_summary": "Very common in GATE. Solve for head loss, discharge, or pipe connection in series/parallel (Q constant vs h_f constant).",
        "typical_mistakes": "Confusing friction coefficient (f') with friction factor (f = 4*f') in Darcy's equation.",
        "memory_aid": "Frictional head loss scales with length and square of velocity.",
        "edges": [
            ("Pipe Flow", "Fluid Mechanics", "part_of"),
            ("Pipe Flow", "Reynolds Number", "depends_on"),
            ("Pipe Flow", "Darcy-Weisbach Equation", "governs")
        ]
    },
    {
        "concept": "Dimensional Analysis",
        "definition": "A method of analyzing relationships between physical quantities by identifying their fundamental dimensions.",
        "formula": "Number of Pi terms = n - m",
        "causes": ["Scaling", "Similarity requirements"],
        "effects": ["Reduction in experimental variables", "Scale-up laws"],
        "dependencies": ["Physical variables", "Fundamental dimensions"],
        "applications": ["Wind tunnel testing", "Hydraulic model design", "Dynamic similarity scaling"],
        "failure_modes": ["Missing critical variables", "Scale effects"],
        "related_concepts": ["Buckingham Pi Theorem", "Dynamic Similarity", "Reynolds Number"],
        "prerequisites": ["Fluid Mechanics"],
        "difficulty": "Easy",
        "revision_1_line": "Grouping physical parameters into dimensionless terms.",
        "revision_5_lines": "Uses Buckingham Pi Theorem.\nn variables, m fundamental dimensions -> n-m Pi groups.\nRepeated variables must include all fundamental dimensions.\nPi groups are dimensionless parameters.\nEnables prototype performance prediction from model tests.",
        "explanation": "Dimensional Analysis helps structure complex fluid problems by reducing the dimensional variable count to a smaller set of dimensionless ratios (like Reynolds, Froude, or Mach numbers).",
        "exam_summary": "GATE tests how to select repeating variables and find the number of Pi terms. Ensure repeating variables do not form a dimensionless group.",
        "typical_mistakes": "Selecting repeating variables that have the same dimension or do not contain all fundamental dimensions.",
        "memory_aid": "n variables - m dimensions = Pi groups.",
        "edges": [
            ("Dimensional Analysis", "Fluid Mechanics", "part_of"),
            ("Dimensional Analysis", "Buckingham Pi Theorem", "governs"),
            ("Dimensional Analysis", "Reynolds Number", "affects")
        ]
    },

    # 3. Heat Transfer
    {
        "concept": "Conduction",
        "definition": "Heat transfer through a stationary medium by molecular activity and electron transport.",
        "formula": "q = -k * dT/dx",
        "causes": ["Temperature gradient", "Thermal conductivity"],
        "effects": ["Temperature drop", "Thermal energy flow"],
        "dependencies": ["Thermal conductivity", "Temperature difference", "Material thickness"],
        "applications": ["Insulation design", "Cooling of microelectronics", "Boiler walls"],
        "failure_modes": ["Thermal cracking", "Insulation degradation"],
        "related_concepts": ["Heat Transfer", "Fourier Law", "Thermal Resistance"],
        "prerequisites": ["Heat Transfer"],
        "difficulty": "Easy",
        "revision_1_line": "Fickian-like heat transfer via molecular collisions.",
        "revision_5_lines": "Governed by Fourier's Law.\nOccurs in solids, liquids, and gases (strongest in solids).\nk is thermal conductivity (W/m-K).\nSteady state conduction is linear in flat plates.\nThermal resistance is analogous to electrical resistance.",
        "explanation": "Conduction is the transfer of heat from more energetic particles of a substance to adjacent less energetic ones as a result of interactions between particles.",
        "exam_summary": "Calculate heat flow through composite walls using thermal resistances in series: R = L/(k*A). Very frequent in GATE.",
        "typical_mistakes": "Forgetting the minus sign in Fourier's law, or confusing slab resistance with radial cylinder resistance: ln(r2/r1)/(2*pi*k*L).",
        "memory_aid": "Conduction = contact transfer.",
        "edges": [
            ("Conduction", "Heat Transfer", "part_of"),
            ("Conduction", "Fourier Law", "governs"),
            ("Conduction", "Thermal Resistance", "depends_on")
        ]
    },
    {
        "concept": "Convection",
        "definition": "Heat transfer between a solid surface and a moving fluid by combined bulk motion and conduction.",
        "formula": "q = h * (T_s - T_inf)",
        "causes": ["Fluid motion", "Temperature difference", "Surface shear"],
        "effects": ["Rapid cooling or heating of surfaces"],
        "dependencies": ["Convective heat transfer coefficient", "Fluid velocity", "Viscosity"],
        "applications": ["Radiators", "Boilers", "Electronic heat sinks"],
        "failure_modes": ["Flow blockage", "Fouling of surfaces"],
        "related_concepts": ["Nusselt Number", "Heat Transfer", "Fluid Mechanics"],
        "prerequisites": ["Heat Transfer", "Fluid Mechanics"],
        "difficulty": "Medium",
        "revision_1_line": "Heat transfer between wall and moving fluid.",
        "revision_5_lines": "Governed by Newton's Law of Cooling.\nh is convective heat transfer coefficient (W/m^2-K).\nClassified as free (natural) or forced convection.\nNatural convection is driven by buoyancy forces.\nForced convection is driven by external means (pump/fan).",
        "explanation": "Convection heat transfer is composed of two mechanisms: energy transfer due to random molecular motion (diffusion) and energy transport by bulk fluid motion (advection).",
        "exam_summary": "GATE tests Nusselt number correlations (Nu = h*L/k) to find h. Also know Grashof, Prandtl, and Reynolds relationships.",
        "typical_mistakes": "Using solid thermal conductivity instead of fluid thermal conductivity in Nu = h*L/k.",
        "memory_aid": "Convection = flow cooling.",
        "edges": [
            ("Convection", "Heat Transfer", "part_of"),
            ("Convection", "Nusselt Number", "depends_on"),
            ("Convection", "Fluid Mechanics", "affects")
        ]
    },
    {
        "concept": "Radiation",
        "definition": "Electromagnetic wave energy emission by a body solely due to its temperature.",
        "formula": "E = epsilon * sigma * A * T^4",
        "causes": ["Absolute temperature", "Atomic excitation"],
        "effects": ["Medium-less energy transport"],
        "dependencies": ["Emissivity", "Surface temperature", "Surface area"],
        "applications": ["Spacecraft cooling", "Solar collectors", "Furnace design"],
        "failure_modes": ["Surface degradation", "Overheating in vacuum"],
        "related_concepts": ["Stefan-Boltzmann Law", "Heat Transfer", "Emissivity"],
        "prerequisites": ["Heat Transfer"],
        "difficulty": "Hard",
        "revision_1_line": "Electromagnetic heat transfer; does not require a medium.",
        "revision_5_lines": "Energy emission proportional to T^4.\nStefan-Boltzmann constant sigma = 5.67e-8 W/m^2-K^4.\nReal surfaces have emissivity epsilon < 1.0.\nUses view factors (shape factors) to model geometry.\nView factor reciprocity relation: A_i * F_ij = A_j * F_ji.",
        "explanation": "Radiation is the thermal energy emitted by matter in the form of electromagnetic waves due to changes in the electron configurations of atoms or molecules.",
        "exam_summary": "GATE asks shape factor algebra and net radiation exchange between black/gray bodies: Q = (E_b1 - E_b2) / Sum(Resistances).",
        "typical_mistakes": "Using Celsius instead of Kelvin, which is catastrophic due to the T^4 exponent.",
        "memory_aid": "Radiation = light heat (waves).",
        "edges": [
            ("Radiation", "Heat Transfer", "part_of"),
            ("Radiation", "Stefan-Boltzmann Law", "governs"),
            ("Radiation", "Emissivity", "depends_on")
        ]
    },
    {
        "concept": "Thermal Resistance",
        "definition": "A material's opposition to the flow of heat, analogous to electrical resistance.",
        "formula": "R_t = L / (k * A)",
        "causes": ["Structural material layout", "Low thermal conductivity"],
        "effects": ["Temperature difference across materials"],
        "dependencies": ["Thickness", "Surface area", "Thermal conductivity"],
        "applications": ["Building insulation", "Thermal interface materials", "Composite wall design"],
        "failure_modes": ["Delamination", "Thermal interface degradation"],
        "related_concepts": ["Conduction", "Heat Transfer", "Heat Flux"],
        "prerequisites": ["Conduction", "Heat Transfer"],
        "difficulty": "Easy",
        "revision_1_line": "Heat analog to electrical resistance: R = dT / Q.",
        "revision_5_lines": "Conduction resistance (plane wall): L/(k*A).\nConduction resistance (cylinder): ln(r2/r1)/(2*pi*k*L).\nConvection resistance: 1/(h*A).\nResistances add in series; inverse sums in parallel.\nHelps solve complex 1D heat transfer easily.",
        "explanation": "Thermal resistance is a simplification of heat transfer equations enabling representation as thermal circuits, matching the structure of electrical circuits.",
        "exam_summary": "Solve composite wall or pipe insulation problems by setting up a network of series/parallel resistances. Very common in GATE.",
        "typical_mistakes": "Using the wrong area (like inlet vs outlet) for radial cylinder resistances, or mixing series and parallel branches.",
        "memory_aid": "Resistance = driving force (dT) / flux (Q).",
        "edges": [
            ("Thermal Resistance", "Conduction", "depends_on"),
            ("Thermal Resistance", "Convection", "depends_on"),
            ("Thermal Resistance", "Heat Transfer", "part_of")
        ]
    },
    {
        "concept": "Heat Exchangers",
        "definition": "Devices designed to transfer heat efficiently between two or more fluids at different temperatures.",
        "formula": "Q = U * A * LMTD",
        "causes": ["Temperature difference between hot and cold fluid streams"],
        "effects": ["Fluid heating and cooling"],
        "dependencies": ["Heat transfer coefficient", "Surface area", "Flow arrangement"],
        "applications": ["Steam condensers", "Car radiators", "Evaporators"],
        "failure_modes": ["Fouling", "Tube erosion", "Thermal stress cracking"],
        "related_concepts": ["Heat Transfer", "Convection", "LMTD"],
        "prerequisites": ["Convection", "Heat Transfer"],
        "difficulty": "Medium",
        "revision_1_line": "Devices transferring heat between hot and cold fluids.",
        "revision_5_lines": "Classified as parallel-flow, counter-flow, or cross-flow.\nCounter-flow is the most thermally efficient.\nLog Mean Temperature Difference (LMTD) matches exponential profile.\nEffectiveness-NTU method is used when exit temperatures are unknown.\nFouling factor increases thermal resistance over time.",
        "explanation": "Heat exchangers facilitate heat transfer between fluids without mixing them. Double-pipe, shell-and-tube, and compact heat exchangers are common configurations.",
        "exam_summary": "Tested every year. Calculate LMTD = (dT1 - dT2) / ln(dT1/dT2). For parallel/counter flow, identify correct temperature differences.",
        "typical_mistakes": "Confusing parallel and counter flow temperature boundary assignments, or using arithmetic mean instead of logarithmic mean.",
        "memory_aid": "Counter-flow is always better than parallel-flow.",
        "edges": [
            ("Heat Exchangers", "Heat Transfer", "part_of"),
            ("Heat Exchangers", "Convection", "depends_on"),
            ("Heat Exchangers", "LMTD", "governs")
        ]
    },
    {
        "concept": "Boiling",
        "definition": "Phase change from liquid to vapor occurring at a solid-liquid interface when the surface temperature exceeds saturation temperature.",
        "formula": "q = h_boil * (T_s - T_sat)",
        "causes": ["High heat flux", "Surface superheat"],
        "effects": ["Bubble generation", "Extremely high heat transfer rate"],
        "dependencies": ["Surface roughness", "Fluid pressure", "Surface superheat"],
        "applications": ["Power plant boilers", "Nuclear reactor cooling", "Refrigerant evaporators"],
        "failure_modes": ["Critical heat flux (CHF) violation", "Dryout"],
        "related_concepts": ["Phase Change", "Heat Transfer", "Condensation"],
        "prerequisites": ["Convection", "Heat Transfer"],
        "difficulty": "Hard",
        "revision_1_line": "Liquid-to-vapor phase change at heated surface.",
        "revision_5_lines": "Classified into pool boiling and forced convection boiling.\nPool boiling curve contains: natural convection, nucleate, transition, and film boiling.\nNucleate boiling is the most efficient region.\nCritical Heat Flux (CHF) is the maximum point on the curve.\nLeidenfrost effect occurs in the film boiling region.",
        "explanation": "Boiling is characterized by the formation of vapor bubbles, which grow and detach from the heated surface. The high convection and bubble agitation result in high heat transfer coefficients.",
        "exam_summary": "GATE tests pool boiling curve landmarks. Know why CHF (Critical Heat Flux) is a critical operating safety limit.",
        "typical_mistakes": "Assuming boiling heat transfer always increases with surface temperature (in transition/film boiling, it drops due to vapor film barrier).",
        "memory_aid": "Nucleate boiling is active bubble cooling; film boiling is vapor insulating.",
        "edges": [
            ("Boiling", "Heat Transfer", "part_of"),
            ("Boiling", "Phase Change", "depends_on"),
            ("Boiling", "Critical Heat Flux", "controls")
        ]
    },
    {
        "concept": "Condensation",
        "definition": "Phase change from vapor to liquid occurring when the temperature of a vapor is reduced below its saturation temperature.",
        "formula": "q = h_cond * (T_sat - T_s)",
        "causes": ["Vapor cooling", "Contact with subcooled surfaces"],
        "effects": ["Liquid film or drop formation", "High latent heat release"],
        "dependencies": ["Surface wettability", "Vapor purity", "Surface temperature"],
        "applications": ["Steam condensers", "Distillation columns", "Refrigerant condensers"],
        "failure_modes": ["Non-condensable gas accumulation", "Film barrier formation"],
        "related_concepts": ["Phase Change", "Boiling", "Heat Transfer"],
        "prerequisites": ["Convection", "Heat Transfer"],
        "difficulty": "Hard",
        "revision_1_line": "Vapor-to-liquid phase change on cold surfaces.",
        "revision_5_lines": "Classified into film condensation and dropwise condensation.\nDropwise condensation is much more efficient (10x higher h).\nFilm condensation creates a thermal resistance liquid layer.\nGoverned by Nusselt's theory for laminar film condensation.\nNon-condensable gases significantly reduce condensation rate.",
        "explanation": "When vapor contacts a surface at a lower temperature, it condenses. Dropwise condensation occurs on non-wetting surfaces where drops roll off, leaving the surface exposed. Film condensation forms a continuous liquid film.",
        "exam_summary": "GATE tests Nusselt film condensation formula parameters (h proportional to L^-0.25). Underline the advantage of dropwise over filmwise.",
        "typical_mistakes": "Forgetting that dropwise condensation is difficult to maintain in industrial practice because surfaces eventually become wetted.",
        "memory_aid": "Dropwise condensation has no film barrier, so it transfers heat much faster.",
        "edges": [
            ("Condensation", "Heat Transfer", "part_of"),
            ("Condensation", "Phase Change", "depends_on"),
            ("Condensation", "Dropwise Condensation", "governs")
        ]
    },

    # 4. Strength of Materials
    {
        "concept": "Elasticity",
        "definition": "The property of a material to return to its original shape and size after the forces causing deformation are removed.",
        "formula": "sigma = E * epsilon",
        "causes": ["Atomic bond stretching under loading"],
        "effects": ["Recoverable deformation"],
        "dependencies": ["Young's Modulus", "Crystal structure", "Temperature"],
        "applications": ["Springs", "Structural frames", "Elastic components"],
        "failure_modes": ["Plastic deformation", "Yielding"],
        "related_concepts": ["Strength of Materials", "Hooke's Law", "Yielding"],
        "prerequisites": ["Strength of Materials", "Stress"],
        "difficulty": "Easy",
        "revision_1_line": "Property of material to recover shape after load removal.",
        "revision_5_lines": "Linear elastic behavior is governed by Hooke's Law.\nYoung's Modulus (E) is slope of stress-strain curve.\nReversible atomic displacement is the root cause.\nLimits at proportional and elastic limit.\nDifferent from plasticity and viscoelasticity.",
        "explanation": "Elasticity represents the ability of solid materials to deform under load and return to their baseline dimensions. This is due to interatomic forces acting like micro-springs.",
        "exam_summary": "Hooke's law is fundamental. Understand elastic constants relationship: E = 2*G*(1+mu) = 3*K*(1-2*mu). Highly tested in GATE.",
        "typical_mistakes": "Confusing elastic limit with proportional limit, or mixing elastic constants equations.",
        "memory_aid": "Elasticity = bounce back.",
        "edges": [
            ("Elasticity", "Strength of Materials", "part_of"),
            ("Elasticity", "Hooke's Law", "governs"),
            ("Elasticity", "Yielding", "controls")
        ]
    },
    {
        "concept": "Bending",
        "definition": "Structural deformation of a beam under lateral loads, causing normal stress along its length.",
        "formula": "M / I = sigma / y = E / R",
        "causes": ["Transverse loads", "Bending moments"],
        "effects": ["Flexural strain", "Normal stress", "Beam deflection"],
        "dependencies": ["Moment of inertia", "Bending moment", "Distance from neutral axis"],
        "applications": ["Bridge girders", "Structural beams", "Machine frames"],
        "failure_modes": ["Plastic hinge formation", "Flexural buckling", "Shear failure"],
        "related_concepts": ["Bending Stress", "Bending Moment", "Strength of Materials"],
        "prerequisites": ["Strength of Materials", "Stress"],
        "difficulty": "Medium",
        "revision_1_line": "Flexural loading creating normal stresses and deflection.",
        "revision_5_lines": "Normal stress is zero at the neutral axis.\nNormal stress varies linearly with distance y from neutral axis.\nTop fibers contract; bottom fibers stretch (for sagging moment).\nGoverned by Euler-Bernoulli beam theory assumptions.\nBending stress sigma = M * y / I.",
        "explanation": "When lateral loads are applied to a beam, it curves. This induces tensile stress on one side of the neutral axis and compressive stress on the other, while shear stress acts across the section.",
        "exam_summary": "GATE frequently tests calculating maximum bending stress or deflection for cantilever/simply supported beams under UDL or point loads.",
        "typical_mistakes": "Using diameter instead of radius for y_max, or calculating the wrong moment of inertia I (bd^3/12).",
        "memory_aid": "Bending stress is max at the outer surface, zero at the center.",
        "edges": [
            ("Bending", "Strength of Materials", "part_of"),
            ("Bending", "Bending Stress", "leads_to"),
            ("Bending", "Euler-Bernoulli Beam", "governs")
        ]
    },
    {
        "concept": "Torsion",
        "definition": "The twisting of a structural member about its longitudinal axis when subjected to torque.",
        "formula": "T / J = shear_stress / r = G * theta / L",
        "causes": ["Applied torque", "Twisting moments"],
        "effects": ["Shear stress", "Angular twist"],
        "dependencies": ["Polar moment of inertia", "Torsional rigidity", "Shaft length"],
        "applications": ["Drive shafts", "Torsion springs", "Propeller shafts"],
        "failure_modes": ["Torsional buckling", "Shear fracture"],
        "related_concepts": ["Torque", "Shear Stress", "Strength of Materials"],
        "prerequisites": ["Strength of Materials", "Torque"],
        "difficulty": "Medium",
        "revision_1_line": "Twisting deformation producing circular shear stress.",
        "revision_5_lines": "Shear stress is zero at the center of the shaft.\nShear stress is maximum at the outer surface (radius r).\nAngle of twist (theta) is proportional to shaft length L.\nPolar moment of inertia J = pi*d^4/32 for solid circular shaft.\nTorsional shear stress tau = T * r / J.",
        "explanation": "Torsion occurs when torque is applied, causing cross-sections to rotate relative to one another. For circular shafts, the shear stress varies linearly from zero at the center to a maximum at the outer surface.",
        "exam_summary": "GATE commonly tests shaft design under torsion and comparison of solid vs hollow shafts (hollow is lighter for same strength).",
        "typical_mistakes": "Using J for bending stress, or using J = pi*d^4/64 (which is I, J is 2*I).",
        "memory_aid": "Torsion twists, creating circular shear stress.",
        "edges": [
            ("Torsion", "Strength of Materials", "part_of"),
            ("Torsion", "Torque", "depends_on"),
            ("Torsion", "Shear Stress", "leads_to")
        ]
    },
    {
        "concept": "Columns",
        "definition": "Vertical structural elements designed primarily to carry compressive loads along their longitudinal axis.",
        "formula": "P_cr = pi^2 * E * I / L_e^2",
        "causes": ["Axial compressive loading", "Structural weight support"],
        "effects": ["Compressive stress", "Buckling"],
        "dependencies": ["Slenderness ratio", "Boundary conditions", "Cross-section"],
        "applications": ["Building pillars", "Truss compression members", "Piston rods"],
        "failure_modes": ["Lateral buckling", "Crushing", "Yielding"],
        "related_concepts": ["Euler Buckling", "Compression", "Strength of Materials"],
        "prerequisites": ["Strength of Materials", "Stress"],
        "difficulty": "Medium",
        "revision_1_line": "Compression members prone to buckling failure.",
        "revision_5_lines": "Slender columns fail due to elastic buckling.\nShort columns fail due to material crushing (yielding).\nSlenderness ratio lambda = L_e / k_radius_of_gyration.\nEuler's formula is valid only for slender columns (lambda > critical).\nEffective length L_e depends on boundary constraints.",
        "explanation": "Columns are compression elements. While short columns can carry loads up to their yield limit, long slender columns fail suddenly due to geometric instability (buckling) at loads much lower than the yielding load.",
        "exam_summary": "GATE tests critical buckling load P_cr under various end conditions. Be ready to compute radius of gyration: k = sqrt(I/A).",
        "typical_mistakes": "Applying Euler's formula to short columns where crushing dominates, or using the wrong moment of inertia I (always use minimum I).",
        "memory_aid": "Always buckle about the axis of minimum moment of inertia.",
        "edges": [
            ("Columns", "Strength of Materials", "part_of"),
            ("Columns", "Euler Buckling", "prerequisite_for"),
            ("Columns", "Stress", "affects")
        ]
    },

    # 5. Machine Design
    {
        "concept": "Fatigue",
        "definition": "The progressive and localized structural damage that occurs when a material is subjected to cyclic loading.",
        "formula": "sigma_a / S_e + sigma_m / S_ut = 1 / FOS",
        "causes": ["Cyclic stress", "Fluctuating loads", "Stress concentration"],
        "effects": ["Crack initiation", "Crack propagation", "Sudden fracture"],
        "dependencies": ["Number of load cycles", "Surface finish", "Stress amplitude", "Mean stress"],
        "applications": ["Transmission shafts", "Automotive suspensions", "Turbine blades"],
        "failure_modes": ["Fatigue fracture", "Catastrophic failure without warning"],
        "related_concepts": ["Soderberg Line", "Goodman Relation", "Endurance Limit"],
        "prerequisites": ["Machine Design", "Stress"],
        "difficulty": "Medium",
        "revision_1_line": "Material degradation and cracking under fluctuating loads.",
        "revision_5_lines": "Characterized by S-N curves (stress vs cycles).\nEndurance limit (S_e) is stress below which life is infinite.\nFatigue failure is brittle in nature even for ductile metals.\nInitiates at stress concentrations (notches, cracks).\nGoodman, Soderberg, and Gerber relations model mean stress effects.",
        "explanation": "Cyclic loading causes micro-cracks to form at stress concentration points. These cracks grow slightly during each cycle until the remaining solid area can no longer bear the load, leading to instant failure.",
        "exam_summary": "Highly tested in GATE Machine Design. Be comfortable with Soderberg (uses yield strength) and Goodman (uses ultimate strength) lines.",
        "typical_mistakes": "Forgetting to apply stress concentration factor K_f to amplitude stress only, or neglecting surface correction factors.",
        "memory_aid": "Fatigue is structural weariness from cyclic stress.",
        "edges": [
            ("Fatigue", "Machine Design", "part_of"),
            ("Fatigue", "Soderberg Line", "governs"),
            ("Fatigue", "Crack Growth", "leads_to")
        ]
    },
    {
        "concept": "Failure Theories",
        "definition": "Criteria used to predict the yielding or fracture of materials under multi-axial stress states.",
        "formula": "sigma_v = sqrt(sigma_1^2 - sigma_1*sigma_2 + sigma_2^2) <= S_y / FOS",
        "causes": ["Complex multi-axial load states"],
        "effects": ["Yielding prediction"],
        "dependencies": ["Yield strength", "Principal stresses", "Material ductility"],
        "applications": ["Pressure vessels", "Drive shafts under bending and torsion", "Bolts"],
        "failure_modes": ["Yielding", "Plastic deformation", "Brittle fracture"],
        "related_concepts": ["Principal Stress", "Mohr's Circle", "Yield Criteria"],
        "prerequisites": ["Strength of Materials", "Stress"],
        "difficulty": "Hard",
        "revision_1_line": "Criteria matching complex stress states to yield strength.",
        "revision_5_lines": "Maximum Principal Stress theory (Rankine) is best for brittle materials.\nMaximum Shear Stress theory (Tresca) is conservative for ductile materials.\nMaximum Distortion Energy theory (Von Mises) is most accurate for ductile materials.\nPlots safe regions as ellipses or polygons on principal stress axes.\nYielding occurs when equivalent stress exceeds yield strength.",
        "explanation": "Material test data is uniaxial, but real parts face multi-axial stress. Failure theories mathematically transform multi-axial stress components into an equivalent stress to compare with uniaxial test strengths.",
        "exam_summary": "GATE frequently tests Von Mises equivalent stress calculation for a shaft under combined bending M and torsion T.",
        "typical_mistakes": "Using Rankine theory for ductile steel components (which leads to unsafe design), or sign errors in principal stresses.",
        "memory_aid": "Von Mises = distortion energy (ductile); Rankine = principal stress (brittle).",
        "edges": [
            ("Failure Theories", "Machine Design", "part_of"),
            ("Failure Theories", "Principal Stress", "depends_on"),
            ("Failure Theories", "Yield Criteria", "prerequisite_for")
        ]
    },
    {
        "concept": "Shafts",
        "definition": "Rotating machine elements, usually of circular cross-section, used to transmit power and torque.",
        "formula": "d^3 = 16 / (pi * tau_max) * sqrt((K_m * M)^2 + (K_t * T)^2)",
        "causes": ["Torque transmission", "Bending loads from gears/pulleys"],
        "effects": ["Power transmission", "Torsional deflection", "Shaft bending"],
        "dependencies": ["Material strength", "Polar moment of inertia", "Bending moment", "Torque"],
        "applications": ["Automobile axles", "Turbine rotors", "Electric motor shafts"],
        "failure_modes": ["Torsional fatigue", "Excessive bending deflection", "Keyway failure"],
        "related_concepts": ["Torsion", "Bending", "Fatigue", "Torque"],
        "prerequisites": ["Machine Design", "Torsion", "Bending"],
        "difficulty": "Medium",
        "revision_1_line": "Rotating power transmission members subject to torsion and bending.",
        "revision_5_lines": "Designed based on strength and torsional rigidity.\nSubjected to combined bending and torsion.\nKeyways cut in shafts create stress concentration, reducing strength.\nCritical speed (whirling) occurs when rotation speed matches natural frequency.\nHollow shafts are more rigid per unit weight than solid shafts.",
        "explanation": "Shafts are the backbones of rotational power systems. Because they support gears and pulleys, they experience both torque (torsion) and transverse forces (bending), requiring design using combined loading theories.",
        "exam_summary": "GATE tests equivalent torque T_e = sqrt(M^2 + T^2) and equivalent bending moment M_e = 0.5 * (M + sqrt(M^2 + T^2)).",
        "typical_mistakes": "Forgetting the shock/fatigue factors K_m and K_t when specified in the question, or swapping M and T.",
        "memory_aid": "Equivalent torque drives shaft shear stress design.",
        "edges": [
            ("Shafts", "Machine Design", "part_of"),
            ("Shafts", "Torsion", "requires"),
            ("Shafts", "Bending", "requires")
        ]
    },
    {
        "concept": "Bearings",
        "definition": "Machine elements that constrain relative motion to only the desired motion and reduce friction between moving parts.",
        "formula": "L_10 = (C / P)^p",
        "causes": ["Load support", "Shaft rotation"],
        "effects": ["Friction reduction", "Wear prevention", "Shaft alignment"],
        "dependencies": ["Lubricant viscosity", "Load magnitude", "Rotating speed"],
        "applications": ["Automotive axles", "Gearboxes", "Electric motors", "Engine crankshafts"],
        "failure_modes": ["Spalling (fatigue pitting)", "Lubrication failure", "Overheating"],
        "related_concepts": ["Lubrication", "Shafts", "Friction"],
        "prerequisites": ["Machine Design"],
        "difficulty": "Medium",
        "revision_1_line": "Supports rotating shafts while minimizing friction.",
        "revision_5_lines": "Classified into sliding contact (journal) and rolling contact (ball/roller).\nJournal bearings operate on hydrodynamic lubrication.\nSommerfeld number is the key dimensionless parameter for journal bearings.\nRolling bearings have low starting friction.\nL_10 life is the life exceeded by 90% of bearings under load.",
        "explanation": "Bearings carry shaft loads while allowing smooth rotation. Hydrodynamic journal bearings support loads on a thin film of pressurized oil created by the shaft rotation. Ball bearings use steel balls to roll rather than slide.",
        "exam_summary": "GATE asks for L_10 life calculations (p=3 for ball, p=10/3 for roller bearings) or Sommerfeld number parameter relationships.",
        "typical_mistakes": "Using life in cycles instead of hours when comparing speeds, or using the wrong exponent p.",
        "memory_aid": "Life varies inversely with the cube of the load (for ball bearings).",
        "edges": [
            ("Bearings", "Machine Design", "part_of"),
            ("Bearings", "Shafts", "controls"),
            ("Bearings", "Friction", "decreases")
        ]
    },
    {
        "concept": "Gears",
        "definition": "Rotating machine parts having cut teeth which mesh with another toothed part to transmit torque and power.",
        "formula": "F_t = P * 1000 / v",
        "causes": ["Meshing interaction", "Torque conversion"],
        "effects": ["Speed reduction", "Torque multiplication"],
        "dependencies": ["Teeth profile", "Module", "Pressure angle", "Material strength"],
        "applications": ["Automotive transmissions", "Differentials", "Industrial gearboxes"],
        "failure_modes": ["Tooth pitting (fatigue)", "Tooth bending fracture", "Scoring"],
        "related_concepts": ["Law of Gearing", "Pitch Point", "Torque Transfer"],
        "prerequisites": ["Machine Design", "Law of Gearing"],
        "difficulty": "Medium",
        "revision_1_line": "Toothed elements for positive power and velocity transmission.",
        "revision_5_lines": "Avoids slip, ensuring an exact velocity ratio.\nModule m = D / T (diameter over number of teeth) must match.\nInvolute teeth are standard (satisfies law of gearing dynamically).\nLewis bending equation governs tooth strength design.\nWear strength is checked using Buckingham's equation.",
        "explanation": "Gears transmit power via direct contact of meshing teeth. The teeth profiles must follow the law of gearing to maintain a constant speed ratio, preventing vibration. Designers check tooth bending (Lewis equation) and surface wear (pitting) limits.",
        "exam_summary": "Tested often in GATE. Know how to apply the Lewis equation: W_t = sigma * b * p_c * y, and module calculations.",
        "typical_mistakes": "Confusing module with pitch, or using wrong diameter values in speed ratio calculations.",
        "memory_aid": "Module is the key gear scaling parameter.",
        "edges": [
            ("Gears", "Machine Design", "part_of"),
            ("Gears", "Law of Gearing", "governs"),
            ("Gears", "Torque Transfer", "leads_to")
        ]
    },
    {
        "concept": "Springs",
        "definition": "Elastic objects used to store mechanical energy and deflect significantly under load.",
        "formula": "k = G * d^4 / (8 * D^3 * n)",
        "causes": ["Compressive or tensile loading", "Displacement"],
        "effects": ["Force control", "Energy absorption", "Vibration isolation"],
        "dependencies": ["Wire diameter", "Coil diameter", "Active coils count", "Shear modulus"],
        "applications": ["Automotive suspensions", "Engine valves", "Mechanical clocks"],
        "failure_modes": ["Creep", "Fatigue cracking", "Solid bottoming"],
        "related_concepts": ["Elasticity", "Vibration Isolation", "Deflection"],
        "prerequisites": ["Machine Design", "Elasticity"],
        "difficulty": "Medium",
        "revision_1_line": "Resilient members designed to deflect and absorb energy.",
        "revision_5_lines": "Helical springs experience torsional shear stress under axial load.\nWahl factor (K_w) accounts for curvature and direct shear.\nSpring rate (k) is force per unit deflection.\nSprings in series: 1/k_eq = 1/k1 + 1/k2.\nSprings in parallel: k_eq = k1 + k2.",
        "explanation": "Springs deform under load and restore their shape when unloaded. The helical spring wire is subjected primarily to torsion when the spring is compressed axially. Wahl's stress factor is used to account for curvature stress concentrations.",
        "exam_summary": "GATE frequently tests springs in series/parallel configurations, and Wahl stress factor calculations.",
        "typical_mistakes": "Forgetting to include active coils n vs total coils (n+2) in spring rate calculations.",
        "memory_aid": "Series springs are soft (stiffness drops); parallel springs are stiff.",
        "edges": [
            ("Springs", "Machine Design", "part_of"),
            ("Springs", "Elasticity", "depends_on"),
            ("Springs", "Vibration Isolation", "used_in")
        ]
    },
    {
        "concept": "Bolted Joints",
        "definition": "Temporary mechanical fasteners used to join two or more parts together.",
        "formula": "F_i = T / (C * d)",
        "causes": ["Tightening torque", "Preload"],
        "effects": ["Clamping force", "Clamping stress"],
        "dependencies": ["Bolt grade", "Thread pitch", "Friction coefficient"],
        "applications": ["Structural steel frames", "Engine cylinder heads", "Flange joints"],
        "failure_modes": ["Thread stripping", "Bolt shear", "Fatigue failure"],
        "related_concepts": ["Machine Design", "Stress Concentration", "Preload"],
        "prerequisites": ["Machine Design"],
        "difficulty": "Hard",
        "revision_1_line": "Fasteners providing clamping force via thread torque.",
        "revision_5_lines": "Preload (initial tension) prevents joint separation.\nBolt stiffness and member stiffness determine load distribution.\nOnly a fraction of external load is felt by preloaded bolts.\nSubject to shear, tension, or combined load.\nCore diameter d_c = 0.84 * nominal_diameter d.",
        "explanation": "Bolted joints clamp parts together. When tightening a bolt, it stretches (preloads). When external load is applied, the bolt and clamped members share the load based on their relative spring constants (stiffnesses).",
        "exam_summary": "GATE tests load sharing between bolt and members: P_bolt = P_preload + [k_b / (k_b + x_m)] * P_ext.",
        "typical_mistakes": "Assuming the bolt carries the full external load (preloading protects the bolt by distributing load to clamped members).",
        "memory_aid": "Preload protects bolts from cyclic fatigue.",
        "edges": [
            ("Bolted Joints", "Machine Design", "part_of"),
            ("Bolted Joints", "Stress Concentration", "causes"),
            ("Bolted Joints", "Preload", "depends_on")
        ]
    },

    # 6. Theory of Machines
    {
        "concept": "Mechanisms",
        "definition": "Assemblies of rigid bodies connected by joints to transmit or modify motion and forces.",
        "formula": "F = 3 * (N - 1) - 2 * P_1 - P_2",
        "causes": ["Input displacement", "Crank rotation"],
        "effects": ["Output motion", "Path tracing"],
        "dependencies": ["Degrees of freedom", "Link lengths", "Joint types"],
        "applications": ["Engine pistons", "Robotic arms", "Car windshield wipers"],
        "failure_modes": ["Link bending", "Joint wear", "Mechanism locking (dead center)"],
        "related_concepts": ["Theory of Machines", "Degrees of Freedom", "Static Equilibrium"],
        "prerequisites": ["Theory of Machines"],
        "difficulty": "Medium",
        "revision_1_line": "Kinematic chains designed to transmit specific output motion.",
        "revision_5_lines": "Grashof's law governs four-bar mechanism rotatability.\nIf s + l <= p + q, at least one link can rotate 360 degrees.\nDegrees of freedom (mobility) is checked using Grubler's criterion.\nLower pairs (sliding, turning) have surface contact.\nHigher pairs (rolling, point contact) have line/point contact.",
        "explanation": "A mechanism consists of links connected by kinematic pairs (joints) that restrict motion. Degrees of freedom (DoF) calculations determine if it is a structure (DoF <= 0) or a mechanism (DoF >= 1).",
        "exam_summary": "Tested every year. Calculate DoF using Kutzbach/Grubler criteria, or identify link rotatability using Grashof's Law.",
        "typical_mistakes": "Forgetting to identify redundant (idle) degrees of freedom or counting higher pairs incorrectly.",
        "memory_aid": "s+l <= p+q means a crank is possible.",
        "edges": [
            ("Mechanisms", "Theory of Machines", "part_of"),
            ("Mechanisms", "Degrees of Freedom", "depends_on"),
            ("Mechanisms", "Static Equilibrium", "requires")
        ]
    },
    {
        "concept": "Velocity Analysis",
        "definition": "The determination of linear and angular velocities of links in a mechanism.",
        "formula": "v = omega * r",
        "causes": ["Driving link rotation", "Input motion"],
        "effects": ["Link velocity vectors", "Acceleration profiles"],
        "dependencies": ["Angular velocity", "Instantaneous centers of rotation"],
        "applications": ["Cam profile design", "Gear train speed analysis", "Linkage kinematic optimization"],
        "failure_modes": ["Extreme inertia forces due to acceleration spikes"],
        "related_concepts": ["Angular Velocity", "Mechanisms", "Instantaneous Center"],
        "prerequisites": ["Mechanisms", "Theory of Machines"],
        "difficulty": "Medium",
        "revision_1_line": "Determining motion velocities of linkage components.",
        "revision_5_lines": "Solved via relative velocity method or instantaneous center (I-center) method.\nNumber of I-centers N = n*(n-1)/2 for n links.\nKennedy's theorem states that three mutual I-centers lie on a straight line.\nCoriolis component of acceleration occurs in sliding links (2 * v * omega).\nVelocity of sliding is parallel to the guide.",
        "explanation": "Velocity analysis tracks link movements. The instantaneous center method simplifies finding velocities by treating the relative motion of any two links as pure rotation about a temporary common point (I-center).",
        "exam_summary": "GATE frequently tests finding velocity of slider or Coriolis acceleration: a_c = 2 * v * omega. Know its direction (rotate velocity vector by 90 deg in direction of omega).",
        "typical_mistakes": "Forgetting the factor of 2 in Coriolis acceleration, or mislocating I-centers.",
        "memory_aid": "Three collinear centers: Kennedy's rule.",
        "edges": [
            ("Velocity Analysis", "Theory of Machines", "part_of"),
            ("Velocity Analysis", "Mechanisms", "depends_on"),
            ("Velocity Analysis", "Coriolis Acceleration", "leads_to")
        ]
    },
    {
        "concept": "Governors",
        "definition": "Automatic feedback devices used to control the mean speed of an engine under varying load conditions.",
        "formula": "h = g / omega^2",
        "causes": ["Engine speed fluctuations", "Centrifugal force"],
        "effects": ["Control of fuel throttle valve", "Engine speed stabilization"],
        "dependencies": ["Centrifugal force", "Governor height", "Sleeve weight"],
        "applications": ["Steam engines", "Diesel generators", "Industrial turbines"],
        "failure_modes": ["Hunting (overshooting)", "Governor instability", "Sleeve friction sticking"],
        "related_concepts": ["Centrifugal Force", "Feedback Control", "Governors"],
        "prerequisites": ["Theory of Machines"],
        "difficulty": "Medium",
        "revision_1_line": "Feedback device regulating fuel input to control mean engine speed.",
        "revision_5_lines": "Watt governor is the simplest centrifugal governor.\nPorter governor adds a central weight to sleeve for higher speeds.\nProell governor places balls on extensions for increased sensitivity.\nHartnell governor is spring-loaded.\nHunting occurs when governor is too sensitive.",
        "explanation": "When engine load drops, speed increases. Centrifugal force pushes governor flyballs outward, raising a sleeve which throttles fuel input, slowing the engine. This forms a mechanical feedback loop.",
        "exam_summary": "GATE asks for governor height h or sensitiveness calculations. Know Porter and Hartnell speed-height relationships.",
        "typical_mistakes": "Forgetting that central sleeve weight carries Porter governor sensitivity, or mixing up speed units (rpm vs rad/s).",
        "memory_aid": "Height is inversely proportional to square of angular velocity.",
        "edges": [
            ("Governors", "Theory of Machines", "part_of"),
            ("Governors", "Centrifugal Force", "depends_on"),
            ("Governors", "Feedback Control", "governs")
        ]
    },
    {
        "concept": "Balancing",
        "definition": "The process of designing or modifying a machine to reduce inertia forces and couples.",
        "formula": "m_b * r_b = m_u * r_u",
        "causes": ["Mass eccentricity", "Unbalanced rotating or reciprocating mass"],
        "effects": ["Centrifugal forces", "System vibrations", "Bearing stress"],
        "dependencies": ["Eccentric mass", "Speed of rotation", "Center of mass position"],
        "applications": ["Automotive wheels", "Engine crankshafts", "Turbine rotors"],
        "failure_modes": ["Excessive machine vibration", "Bearing wear", "Structural fatigue"],
        "related_concepts": ["Centrifugal Force", "Mechanical Vibrations", "Unbalanced Force"],
        "prerequisites": ["Theory of Machines"],
        "difficulty": "Medium",
        "revision_1_line": "Redistributing mass to eliminate dynamic forces and couples.",
        "revision_5_lines": "Static balancing requires center of mass to lie on axis of rotation.\nDynamic balancing requires centrifugal forces and couples to sum to zero.\nReciprocating engines can only be partially balanced.\nHammer blow is the unbalanced vertical force in locomotives.\nTractive force fluctuations are due to reciprocating imbalance.",
        "explanation": "Eccentric rotating masses create rotating centrifugal forces that vibrate the frame. Dynamic balancing adds counterweights at calculated angles and planes to cancel out all forces and couples.",
        "exam_summary": "GATE tests dynamic balancing equations (sum of forces = 0, sum of moments = 0) and partial balancing parameters (fraction c).",
        "typical_mistakes": "Only balancing forces and neglecting couples (moments) in multi-plane shaft systems.",
        "memory_aid": "Dynamic balance needs both forces and moments to equal zero.",
        "edges": [
            ("Balancing", "Theory of Machines", "part_of"),
            ("Balancing", "Centrifugal Force", "depends_on"),
            ("Balancing", "Mechanical Vibrations", "controls")
        ]
    },
    {
        "concept": "Gyroscopes",
        "definition": "Devices consisting of a spinning rotor mounted in gimbals, exhibiting conservation of angular momentum.",
        "formula": "C_g = I * omega * omega_p",
        "causes": ["Spin velocity", "Precession velocity"],
        "effects": ["Gyroscopic couple", "Stabilization forces"],
        "dependencies": ["Moment of inertia", "Spin velocity", "Precession rate"],
        "applications": ["Ship stabilizers", "Aircraft autopilot guidance", "Gyroscopic compasses"],
        "failure_modes": ["Gimbal lock", "Bearing friction drift"],
        "related_concepts": ["Angular Momentum", "Torque", "Precession"],
        "prerequisites": ["Theory of Machines"],
        "difficulty": "Medium",
        "revision_1_line": "Spinning rotor generating stabilizing gyroscopic torque.",
        "revision_5_lines": "Spin axis, precession axis, and active gyroscopic couple axis are mutually perpendicular.\nGyroscopic couple is C = I * omega * omega_p.\nReactive gyroscopic couple acts in the opposite direction.\nStabilizes ships against rolling and pitching.\nAffects automotive steering stability during turns.",
        "explanation": "If a spinning rotor's axis is forced to turn (precess), it responds by exerting a perpendicular torque (gyroscopic couple). This torque can be harnessed for stabilization or navigation.",
        "exam_summary": "GATE tests gyroscopic couple calculations for airplanes pitching/rolling, ships steering, or vehicles taking a turn.",
        "typical_mistakes": "Incorrectly applying the right-hand rule to find the direction of the reactive gyroscopic couple.",
        "memory_aid": "Spin, precess, torque: all three are perpendicular.",
        "edges": [
            ("Gyroscopes", "Theory of Machines", "part_of"),
            ("Gyroscopes", "Torque", "depends_on"),
            ("Gyroscopes", "Angular Momentum", "requires")
        ]
    },

    # 7. Manufacturing
    {
        "concept": "Casting",
        "definition": "A manufacturing process in which a liquid material is poured into a mold containing a hollow cavity and allowed to solidify.",
        "formula": "t = B * (V / A)^n",
        "causes": ["Molten metal pouring", "Temperature drop"],
        "effects": ["Solidified component", "Microstructure formation"],
        "dependencies": ["Mold material", "Pouring temperature", "Solidification rate"],
        "applications": ["Engine blocks", "Valve bodies", "Turbine casing castings"],
        "failure_modes": ["Shrinkage cavities", "Gas porosity", "Cold shut", "Misrun"],
        "related_concepts": ["Chvorinov's Rule", "Solidification Time", "Riser Design"],
        "prerequisites": ["Manufacturing Technology"],
        "difficulty": "Medium",
        "revision_1_line": "Pouring molten metal into a mold to solidify into shape.",
        "revision_5_lines": "Pattern is larger than part to account for metal shrinkage.\nRiser acts as molten metal reservoir to feed shrinkage.\nGating system regulates metal flow to prevent turbulence.\nChvorinov's rule dictates solidification time.\nFluidity of metal affects mold cavity filling.",
        "explanation": "Casting allows production of complex shapes. Pouring must be fast enough to prevent freezing before filling (misrun) but slow enough to avoid sand erosion. Solidification shrinkage is compensated by risers.",
        "exam_summary": "Tested every year in GATE. Riser design (Caine's method, modulus method) and solidification time comparisons are crucial.",
        "typical_mistakes": "Forgetting that shrinkage occurs in three stages: liquid shrinkage, solidification shrinkage (both fed by riser), and solid shrinkage (fed by pattern allowance).",
        "memory_aid": "Riser must solidify last: t_riser > t_casting.",
        "edges": [
            ("Casting", "Manufacturing Technology", "part_of"),
            ("Casting", "Chvorinov's Rule", "depends_on"),
            ("Casting", "Riser Design", "requires")
        ]
    },
    {
        "concept": "Welding",
        "definition": "A fabrication process that joins materials, usually metals, by causing fusion through high heat.",
        "formula": "H = V * I / v",
        "causes": ["Arc heat release", "Solidification of weld pool"],
        "effects": ["Coalescence of metals", "Heat-affected zone (HAZ) creation"],
        "dependencies": ["Welding current", "Voltage", "Travel speed", "Shielding gas"],
        "applications": ["Pressure vessels", "Automobile chassis assembly", "Bridge structures"],
        "failure_modes": ["Porosity", "Weld cracking", "Distortion", "Residual stresses"],
        "related_concepts": ["Manufacturing Technology", "Heat Affected Zone", "Fusion"],
        "prerequisites": ["Manufacturing Technology"],
        "difficulty": "Medium",
        "revision_1_line": "Joining metals by localized heat-induced melting and fusion.",
        "revision_5_lines": "Arc welding uses electrical power to melt base and filler metals.\nHeat input H = V * I / v (J/mm) determines weld quality.\nHeat Affected Zone (HAZ) is the region adjacent to weld that undergoes phase change.\nWelding defects include undercutting, slag inclusion, and lack of fusion.\nShielding gas (argon, CO2) prevents weld oxidation.",
        "explanation": "Welding creates a permanent joint by melting the parent metals at the interface. As the weld pool cools, it solidifies into a continuous joint. The high temperature cycle creates metallurgical changes in the HAZ.",
        "exam_summary": "GATE asks for welding heat input, duty cycle, or joint strength calculations. V-I characteristic curves: I = f(V).",
        "typical_mistakes": "Not accounting for arc efficiency (eta) when calculating net heat input: H_net = eta * V * I / v.",
        "memory_aid": "Slower speed means higher heat input and larger HAZ.",
        "edges": [
            ("Welding", "Manufacturing Technology", "part_of"),
            ("Welding", "Heat Affected Zone", "causes"),
            ("Welding", "Fusion", "requires")
        ]
    },
    {
        "concept": "Machining",
        "definition": "Any of various processes in which a piece of raw material is cut into a desired final shape and size by a controlled material-removal process.",
        "formula": "MRR = f * d * V",
        "causes": ["Cutting force", "Tool-workpiece relative motion"],
        "effects": ["Material removal", "Surface finish generation", "Tool wear"],
        "dependencies": ["Cutting speed", "Feed rate", "Depth of cut", "Tool geometry"],
        "applications": ["Shaft turning", "Gear milling", "Hole drilling"],
        "failure_modes": ["Tool breakage", "Excessive tool wear", "Vibration chatter"],
        "related_concepts": ["Taylor's Tool Life Equation", "Tool Wear", "Cutting Speed"],
        "prerequisites": ["Manufacturing Technology"],
        "difficulty": "Hard",
        "revision_1_line": "Cutting raw material to dimensions using wedges/tools.",
        "revision_5_lines": "Merchant's circle is used to analyze orthogonal cutting forces.\nShear angle determines chip thickness and cutting force.\nMerchant's relation: 2 * phi + beta - alpha = pi/2.\nTool life is governed by Taylor's tool life equation.\nBuilt-up edge (BUE) forms at low speeds, damaging finish.",
        "explanation": "Machining uses a wedge-shaped tool to shear chips off a workpiece. The process generates significant heat and friction, which accelerates tool wear and can induce tensile residual stresses in the finished part.",
        "exam_summary": "Extremely highly tested. Master Merchant's circle equations (cutting force F_c, thrust force F_t) and shear angle phi relationships.",
        "typical_mistakes": "Confusing rake angle (alpha) with clearance angle, or using feed rate in the wrong units (mm/s vs mm/rev).",
        "memory_aid": "Merchant circle maps tool forces to shear planes.",
        "edges": [
            ("Machining", "Manufacturing Technology", "part_of"),
            ("Machining", "Taylor's Tool Life Equation", "depends_on"),
            ("Machining", "Cutting Speed", "requires")
        ]
    },
    {
        "concept": "Metal Forming",
        "definition": "Manufacturing processes where plastic deformation is used to change the shape of metal workpieces.",
        "formula": "sigma_flow = K * epsilon^n",
        "causes": ["Applied pressure exceeding material yield strength"],
        "effects": ["Plastic deformation", "Work hardening"],
        "dependencies": ["Yield strength", "Friction coefficient", "Temperature", "Strain rate"],
        "applications": ["Car body panels (stamping)", "Crankshafts (forging)", "Structural pipes (extrusion)"],
        "failure_modes": ["Die wear", "Material tearing/cracking", "Springback"],
        "related_concepts": ["Elasticity", "Yielding", "Yield Criteria"],
        "prerequisites": ["Manufacturing Technology", "Yielding"],
        "difficulty": "Hard",
        "revision_1_line": "Deforming metals plastically to create shapes.",
        "revision_5_lines": "Classified into hot working (above recrystallization temp) and cold working.\nHot working avoids strain hardening and requires lower forces.\nCold working increases strength and surface finish via strain hardening.\nForging, rolling, extrusion, and drawing are primary forming processes.\nTrue strain epsilon = ln(A0/Af) is used in calculations.",
        "explanation": "Metal forming squeezes metal into dies. Plastic deformation alters grain structures, enhancing mechanical properties. Cold working causes dislocation multiplication, hardening the metal.",
        "exam_summary": "GATE tests rolling calculations (draft, maximum reduction = mu^2 * R) and wire drawing/extrusion stresses.",
        "typical_mistakes": "Using engineering strain instead of true strain in flow stress equations.",
        "memory_aid": "Max reduction in rolling depends on friction squared.",
        "edges": [
            ("Metal Forming", "Manufacturing Technology", "part_of"),
            ("Metal Forming", "Yielding", "requires"),
            ("Metal Forming", "Yield Criteria", "depends_on")
        ]
    },
    {
        "concept": "CNC",
        "definition": "Computer Numerical Control, the automated control of machining tools by means of a computer.",
        "formula": "v_f = f * n",
        "causes": ["Programmed G-codes", "Motor drive signals"],
        "effects": ["Precise tool path movement", "Automated manufacturing"],
        "dependencies": ["Controller accuracy", "Motor resolution", "Slide friction"],
        "applications": ["Automated milling", "Automated turning", "3D printing"],
        "failure_modes": ["Controller error", "Tool collision", "Encoder drift"],
        "related_concepts": ["Machining", "Automation", "G-code"],
        "prerequisites": ["Machining", "Manufacturing Technology"],
        "difficulty": "Easy",
        "revision_1_line": "Automating machine tools using computer program inputs.",
        "revision_5_lines": "Operates using G-codes (preparatory commands) and M-codes (miscellaneous commands).\nUses closed-loop encoder feedback for position control.\nInterpolators generate coordinate tool paths (linear/circular).\nBLU (Basic Length Unit) represents the smallest movement increment.\nReduces lead time and eliminates human error.",
        "explanation": "CNC machines execute programmed coordinates. The controller sends step pulses to motors. Linear and circular interpolators ensure smooth movement along complex curved profiles.",
        "exam_summary": "GATE tests G-code identification (G00: rapid travel, G01: linear feed, G02/G03: circular interpolation) and BLU calculations.",
        "typical_mistakes": "Confusing G02 (clockwise circular) with G03 (counter-clockwise circular interpolation).",
        "memory_aid": "BLU is the resolution of CNC motion.",
        "edges": [
            ("CNC", "Manufacturing Technology", "part_of"),
            ("CNC", "Machining", "controls"),
            ("CNC", "Automation", "depends_on")
        ]
    },

    # 8. Vibrations
    {
        "concept": "Free Vibration",
        "definition": "Oscillatory motion of a system under the action of internal forces after an initial disturbance.",
        "formula": "x(t) = A * cos(omega_n * t + phi)",
        "causes": ["Initial displacement or velocity", "Elastic restoring force"],
        "effects": ["Periodic displacement oscillation", "Energy transfer between kinetic and potential"],
        "dependencies": ["System mass", "Stiffness", "Initial conditions"],
        "applications": ["Tuning forks", "Structural natural frequency tests"],
        "failure_modes": ["Amplitude growth (if negative damping)", "Vibration fatigue"],
        "related_concepts": ["Natural Frequency", "Mechanical Vibrations", "Damping"],
        "prerequisites": ["Mechanical Vibrations"],
        "difficulty": "Easy",
        "revision_1_line": "Undriven oscillation at natural frequency.",
        "revision_5_lines": "Natural frequency omega_n = sqrt(k/m).\nNo external excitation force is present.\nEnergy alternates between spring potential and mass kinetic energy.\nOscillation dies out in real systems due to damping.\nRepresented by second-order homogeneous differential equation.",
        "explanation": "If a spring-mass system is pulled and released, it bounces. In the absence of friction, it oscillates forever at its natural frequency. Viscous damping causes the amplitudes to decay exponentially.",
        "exam_summary": "Tested every year in GATE. Calculate natural frequency for single DoF systems (equivalent stiffness methods).",
        "typical_mistakes": "Forgetting to convert mass units (grams to kg) or using wrong spring stiffness equivalents.",
        "memory_aid": "Free vibration vibrates at its own natural frequency.",
        "edges": [
            ("Free Vibration", "Mechanical Vibrations", "part_of"),
            ("Free Vibration", "Natural Frequency", "depends_on"),
            ("Free Vibration", "Damping", "affects")
        ]
    },
    {
        "concept": "Forced Vibration",
        "definition": "Oscillatory motion of a system driven by a continuous external periodic force.",
        "formula": "X = F_0 / sqrt((k - m * omega^2)^2 + (c * omega)^2)",
        "causes": ["External harmonic force", "Unbalanced rotating mass"],
        "effects": ["Steady-state oscillation at excitation frequency"],
        "dependencies": ["Excitation frequency", "System mass", "Stiffness", "Damping"],
        "applications": ["Shaker tables", "Reciprocating machinery mounts", "Vibratory compaction"],
        "failure_modes": ["Resonance failure", "Severe structural fatigue"],
        "related_concepts": ["Resonance", "Amplitude Decay", "Mechanical Vibrations"],
        "prerequisites": ["Mechanical Vibrations", "Free Vibration"],
        "difficulty": "Medium",
        "revision_1_line": "Oscillation driven continuously by external harmonic force.",
        "revision_5_lines": "Response contains transient and steady-state components.\nTransient response decays to zero due to damping.\nSteady-state response occurs at the excitation frequency.\nMagnification factor measures amplitude amplification.\nControlled by damping near resonance.",
        "explanation": "When an external periodic force acts on a system, it forces it to vibrate at the excitation frequency. The steady-state amplitude depends on the proximity of the excitation frequency to the natural frequency.",
        "exam_summary": "GATE tests steady-state amplitude X and phase angle calculations. Understand magnification factor (MF) curves.",
        "typical_mistakes": "Adding transient and steady-state responses for long-term behavior (transient decays quickly and can be neglected for t -> infinity).",
        "memory_aid": "Forced vibration follows the driver's frequency.",
        "edges": [
            ("Forced Vibration", "Mechanical Vibrations", "part_of"),
            ("Forced Vibration", "Resonance", "leads_to"),
            ("Forced Vibration", "Damping", "depends_on")
        ]
    },
    {
        "concept": "Resonance",
        "definition": "A phenomenon in which a vibrating system responds with maximum amplitude when the frequency of excitation matches its natural frequency.",
        "formula": "omega = omega_n",
        "causes": ["Match of excitation frequency to system natural frequency"],
        "effects": ["Extremely large vibration amplitudes", "High structural stresses"],
        "dependencies": ["Damping ratio", "Natural frequency"],
        "applications": ["Acoustic instruments", "Radio antenna tuning"],
        "failure_modes": ["Tacoma Narrows bridge collapse", "Turbine blade catastrophic fracture"],
        "related_concepts": ["Natural Frequency", "Forced Vibration", "Damping"],
        "prerequisites": ["Forced Vibration", "Mechanical Vibrations"],
        "difficulty": "Easy",
        "revision_1_line": "Extreme vibration when excitation frequency equals natural frequency.",
        "revision_5_lines": "Magnification factor goes to infinity for undamped systems.\nDamping is the only parameter that limits amplitude at resonance.\nPhase angle shifts by 90 degrees at resonance.\nCauses rapid fatigue failure of structures.\nAvoided by tuning natural frequency away from operating speed.",
        "explanation": "At resonance, the excitation force acts in phase with the velocity, continuously feeding energy into the system. Without damping, this energy causes the displacement amplitudes to grow without bound.",
        "exam_summary": "Very common in GATE. Identify resonance conditions (frequency ratio r = 1) and calculate damped resonant amplitude X_max = F_0 / (2*zeta*k).",
        "typical_mistakes": "Believing stiffness controls resonance (at resonance, mass and spring forces cancel, and damping is the sole control parameter).",
        "memory_aid": "Resonance: matching frequencies cause massive shakes.",
        "edges": [
            ("Resonance", "Mechanical Vibrations", "part_of"),
            ("Resonance", "Natural Frequency", "depends_on"),
            ("Resonance", "Damping", "controls")
        ]
    },
    {
        "concept": "Vibration Isolation",
        "definition": "The process of isolating an object from a source of vibrations, characterized by transmissibility.",
        "formula": "TR = sqrt((1 + (2*zeta*r)^2) / ((1 - r^2)^2 + (2*zeta*r)^2))",
        "causes": ["Machine mounting", "Elastomeric isolation pads"],
        "effects": ["Reduction in force transmitted to foundation"],
        "dependencies": ["Frequency ratio r", "Damping ratio"],
        "applications": ["Automotive engine mounts", "Sensitive optical table isolation", "Compressor feet pads"],
        "failure_modes": ["Elastomer degradation", "Resonance amplification during start-up"],
        "related_concepts": ["Transmissibility", "Damping", "Resonance"],
        "prerequisites": ["Mechanical Vibrations", "Damping"],
        "difficulty": "Hard",
        "revision_1_line": "Isolating structures from excitation forces using resilient supports.",
        "revision_5_lines": "Transmissibility TR is ratio of transmitted force to exciting force.\nIsolation occurs only when frequency ratio r > sqrt(2).\nTR > 1 for r < sqrt(2) (amplification region).\nAdding damping reduces peak TR at resonance but increases TR in the isolation region (r > sqrt(2)).\nDesigned using springs and rubber dampers.",
        "explanation": "To prevent heavy machines from shaking their foundations, they are mounted on springs. For isolation to work, the natural frequency of the mount must be much lower than the running speed of the machine (r > 1.414).",
        "exam_summary": "GATE tests transmissibility calculations. Solve for force transmitted to ground: F_t = TR * F_0.",
        "typical_mistakes": "Applying isolation formulas for r < 1.414 (where mounting actually increases transmitted forces compared to a rigid connection).",
        "memory_aid": "Isolate only when r > root 2.",
        "edges": [
            ("Vibration Isolation", "Mechanical Vibrations", "part_of"),
            ("Vibration Isolation", "Transmissibility", "governs"),
            ("Vibration Isolation", "Damping", "requires")
        ]
    },

    # 9. Control Systems
    {
        "concept": "Transfer Functions",
        "definition": "The mathematical representation of a linear time-invariant system relating output to input in Laplace domain under zero initial conditions.",
        "formula": "G(s) = Y(s) / X(s)",
        "causes": ["Input signal excitation"],
        "effects": ["Output signal response", "System characteristic equation"],
        "dependencies": ["System parameters", "Differential equations"],
        "applications": ["Feedback loop control design", "Bode plot analysis", "Stability evaluation"],
        "failure_modes": ["Non-linear behavior failure", "Initial condition mismatch"],
        "related_concepts": ["Control Systems", "Laplace Transform", "Characteristic Equation"],
        "prerequisites": ["Control Systems", "Laplace Transform"],
        "difficulty": "Easy",
        "revision_1_line": "S-domain ratio of output to input for LTI systems.",
        "revision_5_lines": "Assumes all initial conditions are zero.\nFully defines the system dynamics.\nIndependent of input type.\nRoots of denominator are system poles.\nRoots of numerator are system zeros.",
        "explanation": "A transfer function characterizes the input-output behavior of a system. By converting differential equations to algebraic equations in the Laplace s-domain, it simplifies feedback loop analysis.",
        "exam_summary": "GATE tests block diagram reduction and Mason's gain formula to find overall transfer functions: T(s) = Sum(P_k * Delta_k) / Delta.",
        "typical_mistakes": "Neglecting feedback path signs in Mason's gain formula, or applying it to non-linear systems.",
        "memory_aid": "Output/Input in s-domain with zero initial conditions.",
        "edges": [
            ("Transfer Functions", "Control Systems", "part_of"),
            ("Transfer Functions", "Laplace Transform", "depends_on"),
            ("Transfer Functions", "Characteristic Equation", "governs")
        ]
    },
    {
        "concept": "Stability",
        "definition": "The property of a control system to produce bounded outputs in response to bounded inputs.",
        "formula": "Real(s_i) < 0",
        "causes": ["Feedback loop gains", "System parameters"],
        "effects": ["Controlled output tracking", "Convergence of errors"],
        "dependencies": ["Closed-loop pole locations", "Loop gain", "Controller parameters"],
        "applications": ["Flight control guidance", "Chemical process control temperature loops", "Motor speed controllers"],
        "failure_modes": ["Unbounded oscillations", "System thermal runaway", "Actuator saturation damage"],
        "related_concepts": ["Closed Loop Stability", "Routh-Hurwitz Criterion", "Transfer Functions"],
        "prerequisites": ["Control Systems", "Transfer Functions"],
        "difficulty": "Medium",
        "revision_1_line": "All closed-loop poles must lie in left half of s-plane.",
        "revision_5_lines": "BIBO (Bounded-Input Bounded-Output) stability is standard.\nStable systems have transient responses that decay to zero.\nPoles on imaginary axis create marginal stability (oscillations).\nPoles in right-half s-plane create absolute instability.\nChecked using Routh-Hurwitz, Root Locus, or Nyquist criteria.",
        "explanation": "Stability is the most critical control requirement. An unstable system reacts to disturbances by generating growing oscillations, which eventually saturates or breaks the physical hardware.",
        "exam_summary": "GATE frequently asks to determine the stability range of a gain parameter K using the Routh-Hurwitz criterion.",
        "typical_mistakes": "Assuming open-loop stability guarantees closed-loop stability (feedback can easily destabilize a stable open-loop system).",
        "memory_aid": "Stable poles live on the left side of the complex s-plane.",
        "edges": [
            ("Stability", "Control Systems", "part_of"),
            ("Stability", "Closed Loop Stability", "prerequisite_for"),
            ("Stability", "Routh-Hurwitz Criterion", "depends_on")
        ]
    },
    {
        "concept": "Root Locus",
        "definition": "A graphical method showing the paths of closed-loop poles in the s-plane as a system parameter (gain K) varies from zero to infinity.",
        "formula": "1 + K * G(s) * H(s) = 0",
        "causes": ["Feedback gain variation"],
        "effects": ["Pole trajectory tracing", "Stability limit identification"],
        "dependencies": ["Open-loop poles and zeros", "Loop gain K"],
        "applications": ["Controller gain selection", "System stability analysis", "Dominant pole design"],
        "failure_modes": ["High gain instability"],
        "related_concepts": ["Closed Loop Stability", "Transfer Functions", "Poles"],
        "prerequisites": ["Control Systems", "Transfer Functions", "Stability"],
        "difficulty": "Hard",
        "revision_1_line": "Plotting closed-loop pole locations as gain K varies.",
        "revision_5_lines": "Starts at open-loop poles (K=0) and ends at open-loop zeros (K=infinity).\nSymmetric about the real axis.\nAsymptotes intersect real axis at centroid sigma.\nAngle of asymptotes depends on pole-zero difference.\nBreakaway/break-in points found using dK/ds = 0.",
        "explanation": "The Root Locus technique allows engineers to see how changing controller gain shifts the closed-loop poles of the system, helping select a gain that balances speed of response and stability.",
        "exam_summary": "GATE tests root locus construction rules: find centroid, breakaway points, or crossing point with imaginary axis.",
        "typical_mistakes": "Applying breakaway formulas incorrectly, or drawing root locus branches off the real axis without checking angle rules.",
        "memory_aid": "Starts at poles, ends at zeros, symmetric.",
        "edges": [
            ("Root Locus", "Control Systems", "part_of"),
            ("Root Locus", "Closed Loop Stability", "used_in"),
            ("Root Locus", "Transfer Functions", "depends_on")
        ]
    },
    {
        "concept": "Frequency Response",
        "definition": "The steady-state response of a system to a sinusoidal input of varying frequency.",
        "formula": "G(j * omega) = M * e^(j * phi)",
        "causes": ["Sinusoidal input excitation"],
        "effects": ["Output magnitude scaling", "Output phase shifting"],
        "dependencies": ["Excitation frequency", "Loop transfer function"],
        "applications": ["Bode plot design", "Nyquist stability criteria", "Low-pass/high-pass filter design"],
        "failure_modes": ["Gain/phase margin degradation", "Feedback instability"],
        "related_concepts": ["Bode Plot", "Gain Margin", "Phase Margin"],
        "prerequisites": ["Control Systems", "Transfer Functions"],
        "difficulty": "Hard",
        "revision_1_line": "Steady-state response to sinusoidal inputs over frequencies.",
        "revision_5_lines": "Obtained by substituting s = j*omega in transfer function.\nOutput is a sinusoid of same frequency but different amplitude and phase.\nMagnitude ratio M = |G(j*omega)|.\nPhase angle phi = angle(G(j*omega)).\nRepresented on Bode plots (dB magnitude and phase vs log frequency).",
        "explanation": "Frequency response analysis determines how a system filters input signals. High frequency inputs are typically attenuated in mechanical systems due to inertia, making them act as low-pass filters.",
        "exam_summary": "Very important in GATE. Be ready to calculate resonant peak, bandwidth, or read slopes of Bode plots to determine system transfer functions.",
        "typical_mistakes": "Confusing frequency omega (rad/s) with frequency f (Hz), or making log graph calculation errors.",
        "memory_aid": "s = j * omega maps transfer functions to frequency domain.",
        "edges": [
            ("Frequency Response", "Control Systems", "part_of"),
            ("Frequency Response", "Bode Plot", "governs"),
            ("Frequency Response", "Transfer Functions", "depends_on")
        ]
    },

    # 10. Engineering Mathematics
    {
        "concept": "Differential Equations",
        "definition": "Equations that relate one or more functions and their derivatives.",
        "formula": "a_n * d^ny/dx^n + ... + a_0 * y = f(x)",
        "causes": ["Physical law modeling (e.g. F=ma, Fourier Law)"],
        "effects": ["Continuous trajectory profiles", "System dynamics output"],
        "dependencies": ["Initial conditions", "Boundary conditions", "Forcing function"],
        "applications": ["Mechanical vibrations", "Heat conduction equations", "Fluid streamline analysis"],
        "failure_modes": ["Solution divergence", "Numerical integration instability"],
        "related_concepts": ["Systems of ODEs", "Laplace Transform", "Calculus"],
        "prerequisites": ["Engineering Mathematics"],
        "difficulty": "Medium",
        "revision_1_line": "Equations containing derivatives modeling rate of change.",
        "revision_5_lines": "Classified by order (highest derivative) and degree.\nLinear ODEs satisfy superposition principle.\nTotal solution = Complementary Function (CF) + Particular Integral (PI).\nCF represents transient response (forces set to zero).\nPI represents steady-state response under forcing function.",
        "explanation": "Differential equations are the language of physical modeling. They describe how mechanical states (position, temperature, pressure) change over time and space relative to forces and gradients.",
        "exam_summary": "GATE tests solving first-order linear ODEs: dy/dx + P*y = Q (using integrating factor e^Integral(P*dx)), and second-order linear ODEs.",
        "typical_mistakes": "Forgetting the constant of integration when applying initial conditions, or using incorrect CF roots forms.",
        "memory_aid": "Complementary function (transient) + Particular integral (steady-state).",
        "edges": [
            ("Differential Equations", "Engineering Mathematics", "part_of"),
            ("Differential Equations", "Systems of ODEs", "prerequisite_for"),
            ("Differential Equations", "Laplace Transform", "used_in")
        ]
    },
    {
        "concept": "Linear Algebra",
        "definition": "The branch of mathematics concerning linear equations, vector spaces, and matrices.",
        "formula": "A * x = b",
        "causes": ["Multi-variable system modeling"],
        "effects": ["Coordinate transformations", "State projection"],
        "dependencies": ["Matrix dimensions", "Matrix rank", "Determinant", "Trace"],
        "applications": ["Finite element structural analysis", "State-space control systems", "Computer graphics"],
        "failure_modes": ["Singularity (matrix not invertible)", "Ill-conditioned system convergence failure"],
        "related_concepts": ["Matrix", "Eigenvalue", "Cayley-Hamilton Theorem"],
        "prerequisites": ["Engineering Mathematics"],
        "difficulty": "Medium",
        "revision_1_line": "Vector spaces, matrices, and linear equations systems.",
        "revision_5_lines": "Solves A*x = b using Gaussian elimination.\nSystem has unique solution if det(A) != 0.\nEigenvalues satisfy det(A - lambda*I) = 0.\nRank of matrix is number of linearly independent rows/columns.\nDiagonalization simplifies matrix operations using eigenvectors.",
        "explanation": "Linear algebra provides the mathematical structure to solve large sets of simultaneous equations. This is the foundation of numerical methods like Finite Element Analysis (FEA) and computational control.",
        "exam_summary": "Highly tested. Focus on eigenvalue properties (sum equals trace, product equals determinant) and system consistency rules (rank of A vs rank of augmented matrix).",
        "typical_mistakes": "Mixing up eigenvalue properties (e.g. thinking eigenvalues of A^T are different from A—they are the same).",
        "memory_aid": "Determinant is product of eigenvalues; trace is sum.",
        "edges": [
            ("Linear Algebra", "Engineering Mathematics", "part_of"),
            ("Linear Algebra", "Cayley-Hamilton Theorem", "governs"),
            ("Linear Algebra", "Matrix", "depends_on")
        ]
    },
    {
        "concept": "Laplace Transform",
        "definition": "An integral transform that converts a real variable function (time domain) to a complex variable function (s-domain).",
        "formula": "F(s) = integral_0_to_inf(e^(-s * t) * f(t) * dt)",
        "causes": ["s-domain mapping"],
        "effects": ["Transforms differential equations into algebraic equations"],
        "dependencies": ["Function convergence conditions"],
        "applications": ["Control system transfer functions", "Solving linear differential equations", "Signal filters"],
        "failure_modes": ["Integral divergence"],
        "related_concepts": ["Transfer Functions", "Differential Equations"],
        "prerequisites": ["Engineering Mathematics", "Differential Equations"],
        "difficulty": "Medium",
        "revision_1_line": "Converting time domain functions to s-domain integrals.",
        "revision_5_lines": "Linear operator transform.\nL{df/dt} = s * F(s) - f(0) (converts derivatives to multiplication).\nL{unit step} = 1/s.\nL{exponential e^(at)} = 1/(s-a).\nInverse Laplace transform returns system to time domain.",
        "explanation": "The Laplace transform maps time-domain dynamics to the complex s-domain. This simplifies solving linear differential equations by turning calculus (derivatives/integrals) into standard algebra.",
        "exam_summary": "GATE tests solving second-order ODEs using Laplace transforms, or applying initial/final value theorems: lim_{t->0} f(t) = lim_{s->inf} s*F(s).",
        "typical_mistakes": "Applying initial/final value theorems when the poles of s*F(s) lie in the right-half s-plane (the theorems are invalid in this case).",
        "memory_aid": "Laplace turns derivatives into simple multiplications by s.",
        "edges": [
            ("Laplace Transform", "Engineering Mathematics", "part_of"),
            ("Laplace Transform", "Differential Equations", "used_in"),
            ("Laplace Transform", "Transfer Functions", "prerequisite_for")
        ]
    },
    {
        "concept": "Numerical Methods",
        "definition": "Algorithms that use numerical approximation to solve mathematical analysis problems.",
        "formula": "x_new = x - f(x) / f'(x)",
        "causes": ["Analytical insolvability of complex/non-linear equations"],
        "effects": ["Converging sequences", "Approximate solutions"],
        "dependencies": ["Convergence criteria", "Initial guess", "Step size"],
        "applications": ["Finite Element Method (FEM)", "CFD grid solvers", "Roots of non-linear equations"],
        "failure_modes": ["Divergence", "Division by zero", "Truncation error accumulation"],
        "related_concepts": ["Numerical Instability", "CFD", "Iterative Solver"],
        "prerequisites": ["Engineering Mathematics"],
        "difficulty": "Medium",
        "revision_1_line": "Numerical algorithms for solving equations approximately.",
        "revision_5_lines": "Newton-Raphson is a fast root-finding method (quadratic convergence).\nRunge-Kutta (RK4) is used for solving ordinary differential equations.\nTrapezoidal and Simpson's rules are used for numerical integration.\nEuler's method is the simplest first-order ODE solver.\nRounding and truncation errors affect solution accuracy.",
        "explanation": "When physical equations (like Navier-Stokes or complex boundary stress fields) cannot be solved analytically with pencil and paper, numerical methods discretize the equations to compute high-accuracy approximations.",
        "exam_summary": "GATE tests Newton-Raphson iterations, Simpson's 1/3 and 3/8 integration rules, and Trapezoidal rule error bounds.",
        "typical_mistakes": "Using degrees instead of radians in trigonometric function evaluations during Newton-Raphson iterations, or calculating the wrong step size h.",
        "memory_aid": "Newton-Raphson: intercept of tangent with x-axis.",
        "edges": [
            ("Numerical Methods", "Engineering Mathematics", "part_of"),
            ("Numerical Methods", "CFD", "used_in"),
            ("Numerical Methods", "Numerical Instability", "leads_to")
        ]
    },
    {
        "concept": "Probability",
        "definition": "The mathematical branch concerned with the numerical description of how likely an event is to occur.",
        "formula": "P(A | B) = P(B | A) * P(A) / P(B)",
        "causes": ["Uncertainty", "Random processes"],
        "effects": ["Statistical predictions", "Reliability metrics"],
        "dependencies": ["Sample space", "Event definitions", "Independence assumptions"],
        "applications": ["Quality control charting", "Structural reliability engineering", "Safety risk assessment"],
        "failure_modes": ["Incorrect probability model", "Statistical overfitting"],
        "related_concepts": ["Quality Control", "Reliability Engineering"],
        "prerequisites": ["Engineering Mathematics"],
        "difficulty": "Medium",
        "revision_1_line": "Mathematical modeling of random events and uncertainty.",
        "revision_5_lines": "Satisfies Kolmogorov axioms.\nBayes' Theorem models conditional probability updates.\nProbability density function (PDF) represents continuous distributions.\nNormal (Gaussian) distribution represents sum of random variables.\nMean (expectation) and variance measure central tendency and spread.",
        "explanation": "Probability quantifies chance. In engineering, it is used to model manufacturing tolerances (quality control) and predict material failure rates under stochastic loading (reliability engineering).",
        "exam_summary": "GATE tests Bayes' Theorem, Poisson and Exponential distributions (often modeling queueing or component lifetimes), and Binomial distributions.",
        "typical_mistakes": "Confusing independent events with mutually exclusive events, or misidentifying distribution parameters.",
        "memory_aid": "Bayes: Posterior = Likelihood * Prior / Evidence.",
        "edges": [
            ("Probability", "Engineering Mathematics", "part_of"),
            ("Probability", "Quality Control", "used_in"),
            ("Probability", "Reliability Engineering", "leads_to")
        ]
    }
]

def search_existing_nodes(graph: ReasoningGraph, term: str) -> Optional[str]:
    term_normalized = term.strip().lower()
    for node_name, node in graph.nodes.items():
        if node_name.lower() == term_normalized:
            return node_name
        aliases = node.metadata.get("aliases", [])
        synonyms = node.metadata.get("synonyms", [])
        for a in aliases:
            if str(a).strip().lower() == term_normalized:
                return node_name
        for s in synonyms:
            if str(s).strip().lower() == term_normalized:
                return node_name
    return None

def main():
    print("=== POPULATING PRIMARY KNOWLEDGE SOURCES ===")
    
    # 1. Load Graph
    graph_path = Path("data/mechanical_engineering_graph.json")
    if not graph_path.exists():
        print(f"Error: {graph_path} not found.")
        return
    print(f"Loading graph from {graph_path}...")
    graph = ReasoningGraph.load_json(graph_path)
    print(f"Graph loaded: {len(graph.nodes)} nodes, {sum(len(edges) for edges in graph.edges.values())} edges.")

    # 2. Load Concept DB
    concept_db_path = Path("data/mechanical_concepts.json")
    concepts_db = []
    if concept_db_path.exists():
        print(f"Loading concepts database from {concept_db_path}...")
        with open(concept_db_path, "r", encoding="utf-8") as f:
            concepts_db = json.load(f)
            
    existing_concepts = {c["concept"].strip().lower(): i for i, c in enumerate(concepts_db)}

    # 3. Add Category Base Nodes to Graph (if not present)
    base_categories = [
        "Thermodynamics", "Fluid Mechanics", "Heat Transfer", "Strength of Materials",
        "Machine Design", "Theory of Machines", "Manufacturing Technology",
        "Mechanical Vibrations", "Control Systems", "Engineering Mathematics",
        "Energy", "Entropy", "Viscous Flow", "Skin Friction Drag", "Flow Speed",
        "Venturimeter", "Fourier Law", "Thermal Resistance", "Nusselt Number",
        "LMTD", "Phase Change", "Critical Heat Flux", "Dropwise Condensation",
        "Hooke's Law", "Yielding", "Bending Stress", "Euler-Bernoulli Beam",
        "Shear Stress", "Euler Buckling", "Preload", "Crack Growth", "Yield Criteria",
        "Precession", "Coriolis Acceleration", "Feedback Control", "Dropwise Condensation",
        "Riser Design", "Automation", "Natural Frequency", "Transmissibility",
        "Laplace Transform", "Characteristic Equation", "Bode Plot", "Systems of ODEs",
        "Matrix", "Numerical Instability", "Quality Control", "Reliability Engineering"
    ]
    for cat in base_categories:
        if not search_existing_nodes(graph, cat):
            print(f"Adding base category node: {cat}")
            graph.add_node(cat, activation=0.5, metadata={"sources": [SOURCE_NAME]})

    # 4. Process Concepts
    added_count = 0
    updated_count = 0
    
    for item in CONCEPTS_DATA:
        name = item["concept"]
        
        # Format concept db entry
        entry = {
            "concept": name,
            "definition": item["definition"],
            "formula": item["formula"],
            "causes": item["causes"],
            "effects": item["effects"],
            "dependencies": item["dependencies"],
            "applications": item["applications"],
            "failure_modes": item["failure_modes"],
            "related_concepts": item["related_concepts"],
            "prerequisites": item["prerequisites"],
            "difficulty": item["difficulty"],
            "revision_1_line": item["revision_1_line"],
            "revision_5_lines": item["revision_5_lines"],
            "explanation": item["explanation"],
            "exam_summary": item["exam_summary"],
            "typical_mistakes": item["typical_mistakes"],
            "memory_aid": item["memory_aid"]
        }
        
        # Update concept DB
        name_lower = name.strip().lower()
        if name_lower in existing_concepts:
            idx = existing_concepts[name_lower]
            concepts_db[idx].update(entry)
            updated_count += 1
        else:
            concepts_db.append(entry)
            existing_concepts[name_lower] = len(concepts_db) - 1
            added_count += 1
            
        # Add to Graph
        metadata = {
            "description": item["definition"],
            "formula": item["formula"],
            "causes": item["causes"],
            "effects": item["effects"],
            "dependencies": item["dependencies"],
            "applications": item["applications"],
            "failure_modes": item["failure_modes"],
            "difficulty": item["difficulty"],
            "sources": [SOURCE_NAME]
        }
        graph.add_node(name, activation=1.0, metadata=metadata)
        
        # Add the 3 edges (and ensure they use allowed types)
        for src, tgt, rel in item["edges"]:
            # Ensure target exists in graph
            if not search_existing_nodes(graph, tgt):
                graph.add_node(tgt, activation=0.5, metadata={"sources": [SOURCE_NAME]})
            graph.add_edge(src, tgt, weight=1.0, relation=rel, metadata={"source": SOURCE_NAME})

    # 5. Sanity Checks & Quality Rules
    print("\nVerifying graph quality rules...")
    
    # 5.1 Ensure NO isolated nodes
    isolated_nodes = []
    for node_name in graph.nodes:
        has_in = any(node_name in targets for targets in graph.edges.values())
        has_out = len(graph.edges.get(node_name, {})) > 0
        if not has_in and not has_out:
            isolated_nodes.append(node_name)
            
    if isolated_nodes:
        print(f"Warning: Found {len(isolated_nodes)} isolated nodes in graph: {isolated_nodes}")
        for node in isolated_nodes:
            graph.add_edge("Mechanical Engineering", node, relation="part_of")
            print(f" - Fixed: Connected isolated node '{node}' to 'Mechanical Engineering'")
    else:
        print("Success: No isolated nodes found.")
        
    # 5.2 Ensure at least 3 connections for every newly added concept
    for item in CONCEPTS_DATA:
        name = item["concept"]
        in_edges = sum(1 for src in graph.edges if name in graph.edges[src])
        out_edges = len(graph.edges.get(name, {}))
        total_connections = in_edges + out_edges
        if total_connections < 3:
            print(f"Warning: Concept '{name}' has only {total_connections} connections. Adding more fallback edges.")
            # Add fallback connections using allowed relation types
            graph.add_edge(name, "Mechanical Engineering", relation="part_of")
            graph.add_edge("Mechanical Engineering", name, relation="part_of")
            
    # 6. Save files
    print(f"\nSaving updated concepts database to {concept_db_path}...")
    with open(concept_db_path, "w", encoding="utf-8") as f:
        json.dump(concepts_db, f, indent=2)
        
    print(f"Saving updated graph to {graph_path}...")
    graph.save_json(graph_path)
    
    print("\n=== EXPANSION COMPLETED SUCCESSFULLY ===")
    print(f"Added {added_count} new concepts, updated {updated_count} existing concepts in DB.")
    print(f"Final Graph state: {len(graph.nodes)} nodes, {sum(len(edges) for edges in graph.edges.values())} edges.")

if __name__ == "__main__":
    main()
