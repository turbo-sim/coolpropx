#!/usr/bin/env python3
import pytest

# Define the list of tests
tests_list = [
    "test_example_cases.py",
    "test_default_solvers.py",
    "test_custom_solvers.py",
    "test_helmholtz_evaluation.py",
    "test_spinodal_calculation.py",
    "test_phase_diagrams.py",
]

# Run pytest when this python script is executed
# pytest.main(tests_list)
# pytest.main(tests_list + [""])
pytest.main(tests_list + ["-v"])


