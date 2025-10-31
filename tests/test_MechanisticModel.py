from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import sympy as sp
import sympytorch
import torch

from halcyon.Models.MechanisticModel import PyTorchODE
# import os
# import sys
# # Add the current directory to the Python path
# sys.path.append(os.path.abspath('.'))


def test_steady_state_derivateives_zero():
    """
    Test the integration of the model with multiplicative interactions.
    1. Load a predefined SBML model that includes multiplicative interactions.
    2. Read steady-state values from a TSV file to use as initial conditions.
    3. Initialize the model using the SBML file and sets up the initial state.
    4. Perform a forward pass of the model to compute the derivatives at time t=0.
    5. Assert that the computed derivatives are close to zero, indicating that
       the system is indeed at steady state.
    """
    in_folder = Path("tests/4_genes_multplicative_interactions")
    sbml_path = in_folder / 'integration_test_network.xml'
    
    my_model = PyTorchODE.from_sbml(sbml_path, from_gnw=True)
    
    ss_values = pd.read_csv(in_folder / 'integration_test_network_nonoise_wildtype.tsv', sep='\t')
    ss_dict = ss_values.to_dict('records')[0]
    # Generate a tensor of steady-state (SS) values for each batch
    batch_size = 10  # Define the batch size
    ss_tensor = torch.tensor([list(ss_dict.values())] * batch_size, dtype=torch.float32)

    forward_results = my_model.forward(t = torch.tensor([0]), state = ss_tensor)
    
    assert np.allclose(
        forward_results.detach().numpy(), np.array([0, 0, 0, 0 ]), atol=1e-5
    )

@pytest.fixture
def test_model():
    in_folder = Path("tests/4_genes_multplicative_interactions")
    sbml_path = in_folder / "integration_test_network.xml"
    model = PyTorchODE.from_sbml(sbml_path, from_gnw=True)
    return model

@pytest.fixture
def ss_initial_values():
    """Fixture for steady-state initial values."""
    in_folder = Path("tests/4_genes_multplicative_interactions")
    # sbml_path = in_folder / "integration_test_network.xml"
    ss_values = pd.read_csv(
        in_folder / "integration_test_network_nonoise_wildtype.tsv", sep="\t"
    )
    steady_state = torch.tensor(
        [list(ss_values.to_dict("records")[0].values())], dtype=torch.float32
    )
    return steady_state
    

def test_knockout_masks_zero_synthesis_for_target_gene(test_model, ss_initial_values):
    """
    Test the functionality of knockout masks in the mechanistic model to ensure that
    synthesis for the target gene is set to zero when the gene is knocked out.
    This test performs the following steps:
    1. It computes the baseline output of the model with no knockout conditions and
        asserts that the output for the target gene is approximately zero.
    2. It creates a knockout tensor that sets the specified target gene to a knockout
        condition (1.0) while keeping others at zero.
    3. The model is then evaluated again with the knockout condition, and the result
        for the target gene is compared against the expected value calculated from the
        decay dictionary.
    4. It asserts that the output for the knocked-out gene matches the expected value
        and that the outputs for all other genes remain unchanged compared to the baseline.
    5. Finally, it resets the model conditions to their original state.
    Parameters:
    - test_model: An instance of the mechanistic model to be tested.
    - ss_initial_values: Initial steady-state values for the model's genes.
    Assertions:
    - The output for the target gene after knockout should be close to the expected
      decay value.
    - The outputs for all other genes should remain consistent with the baseline output.
    """
   
    baseline = test_model.forward(torch.tensor([0.0]), ss_initial_values).squeeze(0)
    npt.assert_allclose(baseline.numpy(), 0.0, atol=1e-5)

    ko_tensor = torch.zeros(1, test_model.n_genes)
    ko_index = 0
    ko_tensor[0, ko_index] = 1.0
    test_model.set_conditions(ko_tensor)

    ko_result = test_model.forward(torch.tensor([0.0]), ss_initial_values).squeeze(0)
    target_gene = test_model.modelled_gene_labels[ko_index]
    state_dict = {
        gene: float(ss_initial_values[0, idx]) for idx, gene in enumerate(test_model.modelled_gene_labels)
    }
    state_dict["cell"] = 1.0
    subs_dict = test_model.param_dict | state_dict
    expected = -float(test_model.decay_dict[target_gene].subs(subs_dict))
    assert np.isclose(ko_result[ko_index].item(), expected, atol=1e-5)
    npt.assert_allclose(
        ko_result[ko_index + 1 :].detach().numpy(),
        baseline[ko_index + 1 :].detach().numpy(),
        atol=1e-5,
    )
    test_model.reset_conditions()


def test_integrate_handles_batched_inputs(test_model, ss_initial_values):
    """Test that the integrate method can handle batched inputs correctly by checking the output shape."""
    batch_size = 2
    batched_state = ss_initial_values.repeat(batch_size, 1)
    t_eval = torch.linspace(0.0, 1.0, steps=6)
    result = test_model.integrate(batched_state, t_eval)
    assert result.shape == (batch_size, test_model.n_genes, t_eval.numel())


def test_fix_gnw_division_brackets_groups_denominator():
    """Test that the _fix_gnw_division_brackets method correctly groups the denominator in division expressions."""
    formula = "a/b*c"
    fixed = PyTorchODE._fix_gnw_division_brackets(formula)
    assert fixed == "a/(b * c)"
    
def test_fix_gnw_multiple_division_brackets_groups_denominator():
    """Test that the _fix_gnw_division_brackets method correctly groups the denominator in division expressions."""
    formula = "a/b*c + d/e*f*g"
    fixed = PyTorchODE._fix_gnw_division_brackets(formula)
    assert fixed == "a/(b * c)+ d/(e * f * g)"


def test_zero_parameters_leads_to_zero_derivatives(test_model, ss_initial_values):
    """
    Test that setting all model parameters to zero leads to derivatives of zero.
    This test performs the following steps:
    1. Set all parameters in the model's parameter dictionary to zero.
    2. Use steady-state initial values to compute the derivatives.
    3. Assert that the computed derivatives are close to zero, indicating that
        the parameters are being used correctly.
    """
    # Set all parameters to zero
    for key in test_model.param_dict.keys():
        test_model.param_dict[key] = 0.0

    # Compute the derivatives at steady state
    derivatives = test_model.forward(torch.tensor([0.0]), ss_initial_values)

    # Assert that all derivatives are close to zero
    assert np.allclose(derivatives.detach().numpy(), 0.0, atol=1e-5)


def test_forward_compiles_and_uses_torch_ops():
    gene = sp.Symbol("Gene")
    synth_expr = sp.Symbol("alpha_Gene") * (sp.sin(gene) + sp.Max(gene, 0))
    decay_expr = sp.Symbol("beta_Gene") * gene
    params = {"alpha_Gene": 1.5, "beta_Gene": 0.3}
    model = PyTorchODE(["Gene"], {"Gene": synth_expr}, {"Gene": decay_expr}, params)
    assert not model._compiled_ready

    state = torch.tensor([[0.4]], requires_grad=True)
    result = model.forward(torch.tensor([0.0]), state)
    assert model._compiled_ready

    state_clamped = torch.clamp(state, min=0.0)
    expected = params["alpha_Gene"] * (
        torch.sin(state_clamped) + torch.maximum(state_clamped, torch.zeros_like(state_clamped))
    ) - params["beta_Gene"] * state_clamped
    torch.testing.assert_close(result, expected)

    result.sum().backward()
    assert state.grad is not None
    assert isinstance(model._synthesis_module, sympytorch.SymPyModule)
    module_inputs = {
        "Gene": state_clamped[:, 0],
        "cell": torch.ones_like(state_clamped[:, 0]),
    }
    synthesis_output = model._synthesis_module(**module_inputs)
    torch.testing.assert_close(
        synthesis_output, params["alpha_Gene"] * (
            torch.sin(state_clamped) + torch.maximum(state_clamped, torch.zeros_like(state_clamped))
        ),
    )
    assert synthesis_output.device == state.device


def test_compile_expressions_rejects_unknown_sympy_function():
    gene = sp.Symbol("Gene")
    synth_expr = sp.gamma(gene)
    decay_expr = sp.Symbol("delta_Gene") * gene
    params = {"delta_Gene": 0.5}
    model = PyTorchODE(["Gene"], {"Gene": synth_expr}, {"Gene": decay_expr}, params)

    with pytest.raises(ValueError, match="Unsupported sympy function 'gamma'"):
        model.forward(torch.tensor([0.0]), torch.ones(1, 1))
