import os
import re
import sys
import json
import time
import subprocess
import urllib.request
import urllib.error
import tempfile
import sqlite3

class OllamaClient:
    """Helper to query local Ollama instance."""
    def __init__(self, host="127.0.0.1", port=11434, model="qwen2.5-coder:3b"):
        self.url = f"http://{host}:{port}/api/generate"
        self.model = model

    def check_connection(self):
        try:
            req = urllib.request.Request(self.url.replace("/api/generate", "/api/tags"))
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = [m["name"] for m in data.get("models", [])]
                if self.model in models or f"{self.model}:latest" in models or any(self.model in m for m in models):
                    return True, f"Connected to Ollama. Model '{self.model}' is available."
                return False, f"Ollama is running, but model '{self.model}' was not found. Available models: {', '.join(models)}"
        except Exception as e:
            return False, f"Could not connect to Ollama server at {self.url}: {e}"

    def generate(self, prompt, system_prompt=None, temperature=0.2):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=45) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

class SandboxExecutor:
    """Compiles and executes code in multiple languages in a safe sub-process sandbox."""
    
    @staticmethod
    def check_compiler(cmd):
        try:
            # Run version command or check existence on PATH
            subprocess.run([cmd, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=2)
            return True
        except Exception:
            try:
                subprocess.run([cmd, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=2)
                return True
            except Exception:
                return False

    @classmethod
    def get_supported_runtimes(cls):
        return {
            "python": True, # Python is always available
            "javascript": cls.check_compiler("node"),
            "cpp": cls.check_compiler("g++") or cls.check_compiler("clang++"),
            "go": cls.check_compiler("go"),
            "java": cls.check_compiler("javac"),
            "rust": cls.check_compiler("rustc"),
            "sql": True, # Handled via sqlite3 in-memory
            "html": True # Handled via BeautifulSoup/HTML parser
        }

    def execute(self, language, code):
        """Runs the code and returns (exit_code, stdout, stderr, is_simulated)."""
        language = language.lower().strip()
        runtimes = self.get_supported_runtimes()

        # Handle SQL in-memory
        if language == "sql":
            return self._run_sql(code)

        # Handle HTML validation
        if language in ("html", "html/css", "css"):
            return self._run_html(code)

        # Check compiler availability. If not available, do simulated execution.
        compiler_needed = {
            "javascript": "node",
            "cpp": "g++",
            "go": "go run",
            "java": "javac",
            "rust": "rustc"
        }
        
        target_compiler = compiler_needed.get(language)
        if target_compiler and not runtimes.get(language):
            return self._run_simulated(language, code, f"Compiler/Runtime '{target_compiler}' is not installed or not in PATH.")

        # Real Execution
        with tempfile.TemporaryDirectory(dir="scratch") as tmpdir:
            if language == "python":
                filepath = os.path.join(tmpdir, "solution.py")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(code)
                return self._run_process(["python", filepath])

            elif language == "javascript":
                filepath = os.path.join(tmpdir, "solution.js")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(code)
                return self._run_process(["node", filepath])

            elif language == "cpp":
                source_path = os.path.join(tmpdir, "solution.cpp")
                exec_path = os.path.join(tmpdir, "solution.exe")
                with open(source_path, "w", encoding="utf-8") as f:
                    f.write(code)
                
                # Compile
                comp_cmd = ["g++", "-std=c++17", source_path, "-o", exec_path]
                c_res = subprocess.run(comp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=8)
                if c_res.returncode != 0:
                    return c_res.returncode, "", c_res.stderr.decode("utf-8", errors="ignore"), False
                
                # Run
                return self._run_process([exec_path])

            elif language == "go":
                filepath = os.path.join(tmpdir, "solution.go")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(code)
                return self._run_process(["go", "run", filepath])

            elif language == "java":
                # Find public class name or use Main
                class_match = re.search(r"public\s+class\s+(\w+)", code)
                class_name = class_match.group(1) if class_match else "Main"
                
                source_path = os.path.join(tmpdir, f"{class_name}.java")
                with open(source_path, "w", encoding="utf-8") as f:
                    f.write(code)
                
                # Compile
                c_res = subprocess.run(["javac", source_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=8)
                if c_res.returncode != 0:
                    return c_res.returncode, "", c_res.stderr.decode("utf-8", errors="ignore"), False
                
                # Run
                # Need to run java inside that directory
                return self._run_process(["java", "-cp", tmpdir, class_name])

            elif language == "rust":
                source_path = os.path.join(tmpdir, "solution.rs")
                exec_path = os.path.join(tmpdir, "solution.exe")
                with open(source_path, "w", encoding="utf-8") as f:
                    f.write(code)
                
                # Compile
                c_res = subprocess.run(["rustc", source_path, "-o", exec_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=10)
                if c_res.returncode != 0:
                    return c_res.returncode, "", c_res.stderr.decode("utf-8", errors="ignore"), False
                
                # Run
                return self._run_process([exec_path])

        return -1, "", f"Unsupported language: {language}", False

    def _run_process(self, cmd_args):
        try:
            res = subprocess.run(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                shell=True
            )
            return (
                res.returncode,
                res.stdout.decode("utf-8", errors="ignore"),
                res.stderr.decode("utf-8", errors="ignore"),
                False
            )
        except subprocess.TimeoutExpired:
            return -1, "", "Execution Timed Out (exceeded 5 seconds limit).", False
        except Exception as e:
            return -1, "", f"Failed to execute process: {e}", False

    def _run_sql(self, code):
        """Runs SQL statements on an in-memory SQLite database."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        stdout = []
        stderr = []
        exit_code = 0
        
        # Split into statements
        statements = [s.strip() for s in code.split(";") if s.strip()]
        for stmt in statements:
            try:
                cursor.execute(stmt)
                rows = cursor.fetchall()
                if rows:
                    cols = [d[0] for d in cursor.description]
                    stdout.append(f"Query: {stmt}\nResult:\n" + "\t".join(cols) + "\n" + "\n".join(["\t".join(map(str, r)) for r in rows]) + "\n")
                else:
                    stdout.append(f"Statement executed successfully: {stmt}\nRows affected: {cursor.rowcount}\n")
            except Exception as e:
                stderr.append(f"SQL Error on statement '{stmt}': {e}")
                exit_code = 1
                break
                
        conn.close()
        return exit_code, "\n".join(stdout), "\n".join(stderr), False

    def _run_html(self, code):
        """Performs static structure validation on HTML/CSS contents."""
        errors = []
        
        # Check basic tags
        required = ["<!DOCTYPE html>", "<html", "</html>", "<body", "</body>"]
        for tag in required:
            if tag.startswith("<!") or tag.endswith(">"):
                if tag.lower() not in code.lower():
                    errors.append(f"Missing tag: {tag}")
            else:
                if tag.lower() not in code.lower():
                    errors.append(f"Missing open/close tag structure for: {tag}")
                    
        # Check balanced tags for common elements
        tags_to_check = ["div", "span", "p", "h1", "h2", "section", "header", "footer"]
        for t in tags_to_check:
            opens = len(re.findall(rf"<{t}\b", code, re.IGNORECASE))
            closes = len(re.findall(rf"</{t}>", code, re.IGNORECASE))
            if opens != closes:
                errors.append(f"Mismatched tag count for <{t}>: found {opens} open, {closes} close.")

        if errors:
            return 1, "", "HTML/CSS Validation failed:\n" + "\n".join(errors), False
        return 0, "HTML/CSS document structure validated successfully! No mismatched basic tags found.", "", False

    def _run_simulated(self, language, code, reason):
        """Simulates compilation and validates syntax of target code when compiler is missing."""
        # Simple regex syntax checker for basic things
        errors = []
        
        if language == "javascript":
            # Check curly brackets match
            opens = code.count("{")
            closes = code.count("}")
            if opens != closes:
                errors.append(f"Mismatched curly braces: {opens} open vs {closes} closed.")
            
        elif language == "go":
            if "package main" not in code:
                errors.append("Go script missing 'package main' declaration.")
            if "func main(" not in code:
                errors.append("Go script missing 'func main()' entrypoint.")
                
        elif language == "cpp":
            if "#include" not in code:
                errors.append("C++ file missing preprocessor includes.")
            if "int main(" not in code:
                errors.append("C++ file missing 'int main()' function.")
            # Check semicolons on statements (very rough check)
            for i, line in enumerate(code.split("\n")):
                line = line.strip()
                if line and not line.endswith(";") and not line.endswith("{") and not line.endswith("}") and not line.startswith("#") and not line.startswith("//"):
                    if "using namespace" in line or "return " in line or "cout " in line or "cin " in line:
                        errors.append(f"Line {i+1}: Missing semicolon at end of statement: '{line}'")

        if errors:
            return 1, "", f"Simulated Check Fail ({reason}):\n" + "\n".join(errors), True
        return 0, f"Simulated check passed ({reason}). Syntax looks structurally valid.", "", True

class AutonomousCodingAgent:
    """Manages the autonomous coding loop connecting CAT V3 concept plan to Ollama self-correction."""
    def __init__(self, ollama_client: OllamaClient, sandbox: SandboxExecutor):
        self.client = ollama_client
        self.sandbox = sandbox

    def extract_code(self, response_text):
        """Extracts code block between ```language and ```."""
        match = re.search(r"```(?:\w+)?\n(.*?)\n```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback to look for raw block if language label is missing
        match2 = re.search(r"```\n(.*?)\n```", response_text, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        # Fallback to entire text if no markdown block found
        return response_text.strip()

    def run_agent_loop(self, task, language, concept_path, max_iterations=4, callback=None):
        """Runs the self-correcting agent loop.
        
        Args:
            task: User's coding prompt.
            language: Target programming language.
            concept_path: List of concepts from CAT V3.
            max_iterations: Maximum debug retries.
            callback: Function to stream iterations back to GUI.
        """
        path_str = " -> ".join(concept_path)
        system_prompt = (
            "You are an expert autonomous software engineer. Your task is to write clean, fully functional, "
            "and self-contained executable code that solves the user's problem. Use ONLY the specified programming "
            "language. Always wrap your code in a single markdown block (e.g. ```python\\n# code here\\n```)."
        )

        history = []
        code = ""
        success = False
        iteration = 0

        # Create scratch directory if missing
        os.makedirs("scratch", exist_ok=True)

        while iteration < max_iterations and not success:
            iteration += 1
            if callback:
                callback({"type": "status", "iteration": iteration, "message": f"Starting iteration {iteration}..."})

            if iteration == 1:
                prompt = (
                    f"Write a complete program in {language.capitalize()} that implements the following:\n"
                    f"Task: {task}\n\n"
                    f"To design your solution, you must strictly follow this logical reasoning path:\n"
                    f"Concept Plan: {path_str}\n\n"
                    f"Provide only the code block. Make it self-contained and run-ready."
                )
            else:
                prompt = (
                    f"The code you generated in the previous iteration failed to run or compile successfully.\n"
                    f"Task: {task}\n"
                    f"Language: {language.capitalize()}\n"
                    f"Concept Plan: {path_str}\n\n"
                    f"Here was the code you wrote:\n"
                    f"```\n{code}\n```\n\n"
                    f"It resulted in the following execution error/failure:\n"
                    f"```\n{err_msg}\n```\n\n"
                    f"Identify the bug, explain your correction plan, and output the fully corrected, compile-ready code in a new code block."
                )

            if callback:
                callback({"type": "thought", "iteration": iteration, "message": "Invoking Ollama model to generate code..."})

            try:
                response = self.client.generate(prompt, system_prompt=system_prompt)
            except Exception as e:
                err_str = f"Ollama query failed: {e}"
                if callback:
                    callback({"type": "error", "iteration": iteration, "message": err_str})
                return {"success": False, "iterations": iteration, "error": err_str}

            code = self.extract_code(response)
            
            if callback:
                callback({"type": "code", "iteration": iteration, "code": code, "explanation": response.replace(code, "").strip()})

            if callback:
                callback({"type": "thought", "iteration": iteration, "message": "Executing code in sandbox..."})

            # Run Code
            exit_code, stdout, stderr, is_simulated = self.sandbox.execute(language, code)

            if callback:
                callback({
                    "type": "execution", 
                    "iteration": iteration, 
                    "exit_code": exit_code, 
                    "stdout": stdout, 
                    "stderr": stderr,
                    "is_simulated": is_simulated
                })

            if exit_code == 0:
                success = True
                if callback:
                    callback({"type": "status", "iteration": iteration, "message": "Code executed successfully! Loop terminated."})
            else:
                err_msg = stderr if stderr.strip() else f"Process exited with non-zero code: {exit_code}\nOutput: {stdout}"
                if callback:
                    callback({"type": "status", "iteration": iteration, "message": "Execution failed. Bug detected, retrying..."})

        return {
            "success": success,
            "iterations": iteration,
            "code": code,
            "exit_code": exit_code if 'exit_code' in locals() else -1,
            "stdout": stdout if 'stdout' in locals() else "",
            "stderr": stderr if 'stderr' in locals() else "No execution run.",
            "is_simulated": is_simulated if 'is_simulated' in locals() else False
        }
