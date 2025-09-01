#!/usr/bin/env python3
import pytest

# Define the list of tests
tests_list = [
    "test_example_cases.py",
    "test_default_solvers.py",
    "test_custom_solvers.py",
    "test_helmholtz_eos.py",
    "test_spinodal_calculation.py",
    "test_phase_diagrams.py",
<<<<<<< HEAD
    "test_perfect_gas_eos.py",
=======
    # "test_perfect_gas_eos.py",
>>>>>>> 71030d1e03348d1018fc6856af7b1a046a4a12e2
]

# Run pytest when this python script is executed
# pytest.main(tests_list)
# pytest.main(tests_list + [""])
pytest.main(tests_list + ["-v"])


