import unittest
import sys
import os

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_executor import SandboxExecutor, OllamaClient

class TestAgentExecutor(unittest.TestCase):
    def setUp(self):
        self.sandbox = SandboxExecutor()

    def test_python_execution(self):
        code = "print('Hello from Sandbox')"
        exit_code, stdout, stderr, is_sim = self.sandbox.execute("python", code)
        self.assertEqual(exit_code, 0)
        self.assertIn("Hello from Sandbox", stdout)
        self.assertEqual(stderr.strip(), "")
        self.assertFalse(is_sim)

    def test_sql_execution(self):
        code = "CREATE TABLE users (id INT, name TEXT); INSERT INTO users VALUES (1, 'Alice'); SELECT * FROM users;"
        exit_code, stdout, stderr, is_sim = self.sandbox.execute("sql", code)
        self.assertEqual(exit_code, 0)
        self.assertIn("Alice", stdout)
        self.assertEqual(stderr.strip(), "")
        self.assertFalse(is_sim)

    def test_html_validation_success(self):
        code = "<!DOCTYPE html><html><body><div><p>Success</p></div></body></html>"
        exit_code, stdout, stderr, is_sim = self.sandbox.execute("html", code)
        self.assertEqual(exit_code, 0)
        self.assertIn("validated successfully", stdout)
        self.assertFalse(is_sim)

    def test_html_validation_fail(self):
        code = "<html><body><div><p>Mismatched open tags"
        exit_code, stdout, stderr, is_sim = self.sandbox.execute("html", code)
        self.assertEqual(exit_code, 1)
        self.assertIn("HTML/CSS Validation failed", stderr)
        self.assertFalse(is_sim)

    def test_simulated_execution(self):
        code = "package main\nimport \"fmt\"\nfunc main() {\n\tfmt.Println(\"Go Simulated\")\n}"
        # We simulate go execution if compiler is missing or present, so we force compile test to check syntax
        exit_code, stdout, stderr, is_sim = self.sandbox._run_simulated("go", code, "Compiler missing test")
        self.assertEqual(exit_code, 0)
        self.assertIn("Simulated check passed", stdout)
        self.assertTrue(is_sim)

if __name__ == "__main__":
    unittest.main()
