"""Compatibility wrapper for the CAT V2 training CLI.

The old experiment runner trained against random labels and random graphs. CAT
V2 uses supervised reasoning paths, so this wrapper delegates to the canonical
`run_reasoning.py train` command.
"""

from run_reasoning import main


if __name__ == "__main__":
    main(["train"])

