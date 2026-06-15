# grammar_parser.py
import re

WORD_TO_DIGIT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000
}

CONNECTIVE_REPLACEMENTS = {
    r"\bhas\s+a\s+value\s+of\b": " is ",
    r"\bwith\s+a\s+value\s+of\b": " is ",
    r"\bhaving\s+a\s+value\s+of\b": " is ",
    r"\bmeasuring\s+about\b": " is ",
    r"\bmeasures\s+about\b": " is ",
    r"\bmeasures\b": " is ",
    r"\bmeasuring\b": " is ",
    r"\bhaving\b": " is ",
    r"\bequals\b": " is ",
    r"\bequal\s+to\b": " is ",
    r"\bis\s+equal\s+to\b": " is ",
    r"\bvalued\s+at\b": " is ",
    r"\bvalue\s+of\b": " is ",
    r"\bof\s+magnitude\b": " is ",
}

PHRASE_REPLACEMENTS = {
    # Strength of materials & Buckling
    r"\bcross\s*-\s*sectional\s+area\b": "area",
    r"\bsurface\s+area\b": "area",
    r"\baxial\s+load\b": "load",
    r"\bcompressive\s+load\b": "load",
    r"\baxial\s+compressive\s+force\b": "load",
    r"\baxial\s+compressive\s+load\b": "load",
    r"\bcompressive\s+force\b": "load",
    r"\bapplied\s+load\b": "load",
    r"\bapplied\s+force\b": "load",
    r"\byoung\'s\s+modulus\b": "E",
    r"\belastic\s+modulus\b": "E",
    r"\bmodulus\s+of\s+elasticity\b": "E",
    r"\barea\s+moment\s+of\s+inertia\b": "moment of inertia",
    r"\beffective\s+length\b": "effective length",
    r"\beffective\s+column\s+length\b": "effective length",
    
    # Fluids
    r"\bflow\s+speed\b": "speed",
    r"\bflow\s+velocity\b": "speed",
    r"\bfluid\s+velocity\b": "speed",
    r"\bvelocity\s+of\s+flow\b": "speed",
    r"\bfluid\s+density\b": "density",
    r"\bdynamic\s+viscosity\b": "viscosity",
    r"\bpipe\s+diameter\b": "diameter",
    r"\bcharacteristic\s+length\b": "diameter",
    
    # Heat transfer
    r"\bthermal\s+conductivity\b": "k",
    r"\btemperature\s+gradient\b": "gradient",
    r"\bheat\s+flux\b": "q",
    r"\bconvective\s+heat\s+transfer\s+coefficient\b": "h",
    r"\bcold\s+reservoir\s+temperature\b": "T_C",
    r"\bhot\s+reservoir\s+temperature\b": "T_H",
    r"\bcarnot\s+efficiency\b": "efficiency",
    r"\bnusselt\s+number\b": "Nu",
    
    # Mechanics
    r"\bbending\s+moment\b": "M",
    r"\bangular\s+acceleration\b": "alpha",

    # GATE boundary conditions
    r"\bpinned\s+at\s+both\s+ends\b": "pinned ends",
    r"\bboth\s+ends\s+pinned\b": "pinned ends",
    r"\bfixed\s+at\s+both\s+ends\b": "fixed fixed ends",
    r"\bboth\s+ends\s+fixed\b": "fixed fixed ends",
    r"\bone\s+end\s+fixed\s+and\s+the\s+other\s+free\b": "fixed free ends",
    r"\bfixed\s+free\b": "fixed free ends",
    r"\bcantilever\b": "fixed free column",
    r"\bone\s+end\s+fixed\s+and\s+the\s+other\s+pinned\b": "fixed pinned ends",
    r"\bfixed\s+pinned\b": "fixed pinned ends",
}

def words_to_digits_in_text(text: str) -> str:
    # Replace hyphens only in letter-compounds (e.g. cross-sectional), not in numbers/units
    text = re.sub(r"(?<=[a-zA-Z])-(?=[a-zA-Z])", " ", text)
    
    # Match compound words e.g., "twenty five"
    tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    ones = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    
    for t in tens:
        for o in ones:
            compound = f"{t} {o}"
            val = WORD_TO_DIGIT[t] + WORD_TO_DIGIT[o]
            text = re.sub(rf"\b{compound}\b", str(val), text, flags=re.IGNORECASE)
            
    # Match single numbers (from longest to shortest word to prevent partial matches)
    for word in sorted(WORD_TO_DIGIT.keys(), key=len, reverse=True):
        val = WORD_TO_DIGIT[word]
        # Handle X hundred, X thousand
        text = re.sub(rf"\b([0-9]+)\s+{word}\b", lambda m: str(float(m.group(1)) * val), text, flags=re.IGNORECASE)
        # Handle word itself
        text = re.sub(rf"\b{word}\b", str(val), text, flags=re.IGNORECASE)
        
    return text

def normalize_query(text: str) -> str:
    # 1. Convert spelling numbers to digits
    text = words_to_digits_in_text(text)
    
    # 2. Standardize technical synonyms
    for phrase, rep in PHRASE_REPLACEMENTS.items():
        text = re.sub(phrase, rep, text, flags=re.IGNORECASE)
        
    # 3. Standardize connective verbs
    for conn, rep in CONNECTIVE_REPLACEMENTS.items():
        text = re.sub(conn, rep, text, flags=re.IGNORECASE)
        
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
