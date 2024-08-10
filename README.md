# WitnessInformation
Code for the "Information theoretical perspective on the method of Entanglement Witnesses" paper.

https://arxiv.org/abs/2308.07744

## Files:
- **simulation.py** is the main simulation high level logic.
- **random_generation_utils.py** contains functions related to random generation.
- **criterion_utils.py** contains information theoretical related functions and entanglement criteria.
- **test_...** contains tests for the util functions and can be run with pytest.
- **plots_notebook.ipynb** contains all plotting functions and table data calculations.


The cpp_implementation_and_extension folder contains a C++ more efficient implementation of the simulation, approximately an order of magnitude faster. This implementation is still untested.
