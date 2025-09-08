import libsbml
import torch
from halcyon.RegulatoryNetwork import RegulatoryNetwork
from halcyon.Models.Model import Model
from halcyon.Models.MechanisticModel import ComplexODEModel, PyTorchODE
from sympy import symbols, sympify
import networkx as nx
import pytest
import numpy as np


@pytest.fixture
def one_equation_synthesis():
    reader = libsbml.SBMLReader()
    document = reader.readSBML("tests/6_gene_testv2/test_scalefree.xml")  # type: ignore
    model = document.getModel()
    reaction = model.getReaction(0)
    return reaction


@pytest.fixture
def one_equation_degradation():
    reader = libsbml.SBMLReader()
    document = reader.readSBML("tests/6_gene_testv2/test_scalefree.xml")  # type: ignore
    model = document.getModel()
    reaction = model.getReaction(1)
    return reaction


def test_add_one_equation(one_equation_synthesis, one_equation_degradation):
    test_pytorch_ode = PyTorchODE(modelled_gene_labels=[])
    _, sympy_formula_synthesis, synthesis_parameters = (
        test_pytorch_ode.parse_one_reaction(one_equation_synthesis)
    )

    ss_dict = {
        "Node_0": 0.1780818,
        "Node_1": 0.9222381,
        "Node_2": 0.1129278,
        "Node_3": 1.0,
        "Node_4": 0.1310036,
        "Node_5": 0.8210416,
        "cell": 1,
    }

    net_synthesis = sympy_formula_synthesis.subs(synthesis_parameters | ss_dict)
    print(net_synthesis)

    _, sympy_formula_degradation, degradation_parameters = (
        test_pytorch_ode.parse_one_reaction(one_equation_degradation)
    )
    net_degradation = sympy_formula_degradation.subs(synthesis_parameters | ss_dict)
    print(net_degradation)

    assert np.allclose(net_synthesis + net_degradation, 0, atol=1e-5)


def test_import_from_sbml():
    sbml_path = "tests/6_gene_testv2/test_scalefree.xml"
    my_model = PyTorchODE.from_sbml(sbml_path)
    
    ss_dict = {
        "Node_0": 0.1780818,
        "Node_1": 0.9222381,
        "Node_2": 0.1129278,
        "Node_3": 1.0,
        "Node_4": 0.1310036,
        "Node_5": 0.8210416,
        "cell": 1,
    }
    for node_name in my_model.modelled_gene_labels:
        synthesis_formula = my_model.synthesis_dict[node_name]
        degradation_formula = my_model.decay_dict[node_name]

        synthesis_value = synthesis_formula.subs(my_model.param_dict | ss_dict)
        degradation_value = degradation_formula.subs(my_model.param_dict | ss_dict)
        print(node_name, synthesis_value, degradation_value)
        net = synthesis_value - degradation_value
        assert np.allclose(net, 0, atol=1e-5)

    forward_results = my_model.forward(torch.tensor([0]), steady_state_values)

    assert np.allclose(
        forward_results.detach().numpy(), np.array([0, 0, 0, 0, 0]), atol=1e-5
    )


#     time_points = np.array([ 0.        ,  0.97959184,  1.95918367,  2.93877551,  3.91836735,
#         4.89795918,  5.87755102,  6.85714286,  7.83673469,  8.81632653,
#         9.79591837, 10.7755102 , 11.75510204, 12.73469388, 13.71428571,
#        14.69387755, 15.67346939, 16.65306122, 17.63265306, 18.6122449 ,
#        19.59183673, 20.57142857, 21.55102041, 22.53061224, 23.51020408,
#        24.48979592, 25.46938776, 26.44897959, 27.42857143, 28.40816327,
#        29.3877551 , 30.36734694, 31.34693878, 32.32653061, 33.30612245,
#        34.28571429, 35.26530612, 36.24489796, 37.2244898 , 38.20408163,
#        39.18367347, 40.16326531, 41.14285714, 42.12244898, 43.10204082,
#        44.08163265, 45.06122449, 46.04081633, 47.02040816, 48.        ])

#     right_answer = [[         0,         0,         0,         0,           0,         0],
#   [ 0.0281242, 0.0398251, 0.0246195, 0.0403485, 0.000854956,  0.021094],
#   [ 0.0548445, 0.0780641, 0.0486327, 0.0790689,  0.00193319, 0.0410631],
#   [ 0.0802553,   0.11478, 0.0720518,  0.116227,  0.00338495, 0.0600007],
#   [  0.104477,  0.150033, 0.0948728,  0.151886,  0.00531104, 0.0779898],
#   [  0.127661,  0.183878,  0.117041,  0.186106,  0.00777437, 0.0951022],
#   [  0.149984,  0.216367,  0.138393,  0.218945,    0.010805,    0.1114],
#   [  0.171628,  0.247544,  0.158586,   0.25046,   0.0144019,  0.126936],
#   [  0.192741,  0.277447,  0.177072,  0.280703,   0.0185316,  0.141756],
#   [  0.213396,   0.30611,  0.193194,  0.309725,   0.0231285,    0.1559],
#   [  0.233564,  0.333558,  0.206408,  0.337577,   0.0281005,  0.169404],
#   [  0.253121,  0.359816,  0.216513,  0.364304,   0.0333414,  0.182305],
#   [  0.271898,  0.384906,  0.223704,  0.389954,   0.0387468,  0.194638],
#   [  0.289727,  0.408852,  0.228438,  0.414568,   0.0442261,  0.206438],
#   [  0.306473,  0.431678,   0.23125,  0.438189,   0.0497079,  0.217739],
#   [  0.322042,   0.45341,  0.232625,  0.460858,    0.055139,  0.228576],
#   [  0.336381,  0.474076,  0.232955,  0.482611,   0.0604814,  0.238981],
#   [  0.349469,  0.493709,  0.232534,  0.503487,    0.065709,  0.248984],
#   [  0.361313,  0.512343,  0.231577,   0.52352,    0.070804,  0.258615],
#   [  0.371939,  0.530014,  0.230236,  0.542746,   0.0757549,  0.267899],
#   [  0.381388,   0.54676,  0.228624,  0.561195,   0.0805544,  0.276862],
#   [  0.389708,  0.562622,   0.22682,    0.5789,   0.0851984,  0.285526],
#   [  0.396957,  0.577642,  0.224882,  0.595891,   0.0896849,  0.293912],
#   [  0.403193,   0.59186,  0.222852,  0.612196,   0.0940136,   0.30204],
#   [  0.408481,  0.605318,  0.220763,  0.627843,   0.0981854,  0.309926],
#   [  0.412882,  0.618058,  0.218637,  0.642859,    0.102202,  0.317588],
#   [  0.416458,  0.630119,  0.216492,  0.657269,    0.106066,   0.32504],
#   [  0.419271,  0.641541,  0.214342,  0.671098,    0.109779,  0.332297],
#   [  0.421379,  0.652362,  0.212196,  0.684369,    0.113346,   0.33937],
#   [  0.422838,  0.662617,  0.210064,  0.697104,    0.116769,  0.346272],
#   [  0.423702,  0.672342,  0.207951,  0.709325,    0.120051,  0.353012],
#   [  0.424022,  0.681567,  0.205861,  0.721054,    0.123197,  0.359601],
#   [  0.423845,  0.690325,  0.203798,  0.732309,     0.12621,  0.366049],
#   [  0.423217,  0.698644,  0.201764,   0.74311,    0.129094,  0.372362],
#   [   0.42218,  0.706551,  0.199763,  0.753475,    0.131852,   0.37855],
#   [  0.420774,  0.714072,  0.197795,  0.763422,    0.134488,  0.384618],
#   [  0.419034,  0.721231,  0.195862,  0.772967,    0.137006,  0.390574],
#   [  0.416996,  0.728049,  0.193964,  0.782128,    0.139409,  0.396423],
#   [   0.41469,  0.734548,  0.192103,  0.790918,    0.141701,  0.402171],
#   [  0.412147,  0.740746,  0.190277,  0.799355,    0.143885,  0.407824],
#   [  0.409393,  0.746662,  0.188488,   0.80745,    0.145966,  0.413385],
#   [  0.406454,  0.752313,  0.186735,  0.815219,    0.147945,   0.41886],
#   [  0.403352,  0.757714,  0.185018,  0.822675,    0.149827,  0.424252],
#   [   0.40011,  0.762881,  0.183337,   0.82983,    0.151615,  0.429565],
#   [  0.396747,  0.767825,  0.181692,  0.836696,    0.153311,  0.434802],
#   [  0.393281,  0.772561,  0.180082,  0.843285,     0.15492,  0.439967],
#   [  0.389729,  0.777099,  0.178508,  0.849608,    0.156444,  0.445062],
#   [  0.386106,  0.781451,  0.176967,  0.855676,    0.157886,   0.45009],
#   [  0.382425,  0.785628,  0.175461,  0.861499,    0.159248,  0.455054],
#   [    0.3787,  0.789638,  0.173988,  0.867088,    0.160535,  0.459956]]

#     output = my_import_class.integrate([0, 0, 0, 0, 0, 0])
#     assert np.allclose( output, right_answer, atol=1e-5)

# Define parameter dictionary
param_dict = {
    "pSPA": 1.0,
    "dSPA": 0.1,
    "assCOP1SPA1": 0.2,
    "dissCOP1SPA1": 0.05,
    "pCOP1": 1.0,
    "dCOP1": 0.1,
    "iCOP1_light": 0.1,
    "iCOP1_drought": 0.1,
    "iCOP1_heat_shoot": 1.0,
    "pCOP1_heat_shoot": 1.0,
    "iCOP1_SPA1_auto": 0.1,
}

# Define ODEs as lambdas that take the full state‐dict
ODE_dict = {
    "spa1_shoot": lambda s: (
        param_dict["pSPA"]
        - param_dict["dSPA"] * s["spa1_shoot"]
        - param_dict["assCOP1SPA1"] * s["cop1_shoot"] * s["spa1_shoot"]
        + param_dict["dissCOP1SPA1"] * s["cop1_spa1_shoot"]
    ),
    "cop1_shoot": lambda s: (
        param_dict["pCOP1"] * param_dict["pCOP1_heat_shoot"] ** s["heat"]
        - param_dict["dCOP1"] * s["cop1_shoot"]
        - (
            param_dict["iCOP1_light"] * s["light"]
            + param_dict["iCOP1_drought"] * s["drought"]
        )
        * param_dict["iCOP1_heat_shoot"] ** s["heat"]
        * s["cop1_shoot"]
        - param_dict["assCOP1SPA1"] * s["cop1_shoot"] * s["spa1_shoot"]
        + param_dict["dissCOP1SPA1"] * s["cop1_spa1_shoot"]
    ),
    "cop1_spa1_shoot": lambda s: (
        param_dict["assCOP1SPA1"] * s["cop1_shoot"] * s["spa1_shoot"]
        - param_dict["dissCOP1SPA1"] * s["cop1_spa1_shoot"]
        - param_dict["iCOP1_SPA1_auto"] * s["cop1_spa1_shoot"] ** 2
    ),
}

input_dict = {
    "light": lambda t: (t % 24 <= 16),  # light on for 16 hours
    "heat": lambda t: 0,  # no heat stress
    "drought": lambda t: 0,  # no drought stress
}


# test complex ODE mechanistic model
@pytest.fixture
def complex_model():
    return ComplexODEModel(
        ODE_dict=ODE_dict,
        input_dict=input_dict,
        param_dict={
            k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))
            for k, v in param_dict.items()
        },
    )


def test_forward(complex_model):
    t = torch.tensor(0.0)
    state = torch.tensor([1.0, 1.0, 1.0])
    derivatives = complex_model.forward(t, state)
    assert derivatives.shape == state.shape
    assert torch.is_tensor(derivatives)


def test_build_network_from_self(complex_model):
    net = complex_model._build_network_from_self()
    assert isinstance(net, nx.DiGraph)
    assert len(net.nodes) == len(ODE_dict) + len(input_dict)
    assert len(net.edges) > 0


def test_regulatory_network(complex_model):
    regulatory_network = complex_model.regulatory_network
    assert isinstance(regulatory_network, RegulatoryNetwork)
