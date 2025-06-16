from halcyon.Models.MechanisticModel import SBMLImportClassDummy


def test_import_from_sbml():
    sbml_path = "tests/genenetweaver_outputs/test_scalefree.xml"
    my_importclass_test = SBMLImportClassDummy.from_sbml(sbml_path)
