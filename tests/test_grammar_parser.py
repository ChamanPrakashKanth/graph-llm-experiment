import json
import sys
import unittest
from http.server import HTTPServer
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grammar_parser import normalize_query
from gate_router import (
    hop_concepts,
    match_gate_bank,
    process_gate_query,
    route_concepts,
    seed_concepts_from_text,
)
from equations_database import check_and_solve, check_and_solve_chain, extract_variables


class TestGrammarParser(unittest.TestCase):
    def test_spelled_numbers(self):
        self.assertIn("10 kN", normalize_query("ten kN"))
        self.assertIn("5 m^2", normalize_query("five m^2"))
        self.assertEqual(normalize_query("twenty five"), "25")

    def test_connector_normalization(self):
        self.assertIn("force is 10 kN", normalize_query("force equals ten kN"))
        self.assertIn("area is 5", normalize_query("area having a value of five"))

    def test_boundary_conditions(self):
        self.assertIn("pinned ends", normalize_query("pinned at both ends"))
        self.assertIn("fixed free", normalize_query("cantilever column"))


class TestGateRouter(unittest.TestCase):
    def test_concept_seeding(self):
        seeds = seed_concepts_from_text("calculate critical buckling load for pinned column")
        concepts = [s[0] for s in seeds]
        self.assertIn("Euler Buckling", concepts)

    def test_multi_hop_routing(self):
        path = hop_concepts(["Euler Buckling"])
        self.assertIn("Boundary Condition", path)
        self.assertIn("Critical Load", path)

    def test_route_fuses_model_path(self):
        routing = route_concepts(
            "buckling column pinned ends",
            model_path=["Compression", "Buckling"],
            top_concepts=[{"name": "Euler Buckling", "activation": 0.9}],
        )
        self.assertIn("Euler Buckling", routing["concept_path"])
        self.assertIn("Euler Buckling", routing["ranked_equations"])

    def test_gate_bank_match_mcq(self):
        match = match_gate_bank(
            "For a column of length L, if one end is fixed and the other is free, what is the effective length?"
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["source"], "gate_mcq")
        self.assertEqual(match["answer"], "C")

    def test_process_stress_query(self):
        result = process_gate_query(
            "calculate stress if load equals ten kN and area measures five m^2"
        )
        self.assertIsNotNone(result["solved"])
        self.assertEqual(result["solved"]["solved_value"], "2000.0000")
        self.assertIn("Concept Routing Path", result["composed_answer"])


class TestSolverIntegration(unittest.TestCase):
    def test_stress_from_verbal_input(self):
        question = "calculate stress if load equals ten kN and area measures five m^2"
        result = check_and_solve(question)
        self.assertIsNotNone(result)
        self.assertEqual(result["equation"], "Stress")
        self.assertEqual(result["solved_value"], "2000.0000")

    def test_buckling_with_boundary_inference(self):
        question = (
            "A steel column of length 2.0 m has pinned ends. "
            "If E = 200e9 Pa and I = 1.0e-5 m^4, what is the critical buckling load in kN?"
        )
        result = check_and_solve_chain(
            question,
            concept_path=["Euler Buckling"],
            ranked_equations=["Euler Buckling"],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["equation"], "Euler Buckling")
        self.assertIn("L_e", result["inputs"])
        self.assertAlmostEqual(float(result["solved_value"]), 4934.8, delta=50)

    def test_reynolds_numerical(self):
        question = (
            "Calculate the Reynolds number for water (density = 1000 kg/m^3, "
            "viscosity = 0.001 Pa-s) flowing at 2.0 m/s in a 0.05 m diameter pipe."
        )
        result = check_and_solve_chain(question, ranked_equations=["Reynolds Number"])
        self.assertIsNotNone(result)
        self.assertEqual(result["equation"], "Reynolds Number")
        self.assertAlmostEqual(float(result["solved_value"]), 100000.0, delta=1)


class TestChatServerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from chat_server import ChatRequestHandler

        cls.server = HTTPServer(("127.0.0.1", 0), ChatRequestHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_chat_endpoint_gate_stress(self):
        import urllib.request

        payload = json.dumps({
            "question": "calculate stress if load equals ten kN and area measures five m^2",
            "domain": "mechanical_engineering",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.assertIn("solved_equation", data)
        self.assertEqual(data["solved_equation"]["solved_value"], "2000.0000")
        self.assertIn("gate_routing", data)

    def test_gate_suggestions_endpoint(self):
        import urllib.request

        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/api/gate/suggestions", timeout=10
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.assertGreater(len(data), 5)


if __name__ == "__main__":
    unittest.main()
