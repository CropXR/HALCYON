from mlmmint.ExpressionData import ExpressionData
from mlmmint.Models.HybridModel import HybridModel
from mlmmint.Models.Model import Model
from mlmmint.RegulatoryNetwork import RegulatoryNetwork


class Evaluator:
    def __init__(self, gt_network: RegulatoryNetwork, gt_expressions: ExpressionData):
        self.gt_network = gt_network
        self.gt_expressions = gt_expressions

    def compare_edge_jaccard_gt_vs_inferred_network(
        self, fitted_model: HybridModel | Model
    ):
        inferred_network = fitted_model.get_inferred_network()
        inferred_edges_set = set(inferred_network.get_all_interactions(with_data=False))
        gt_edges_set = set(self.gt_network.get_all_interactions(with_data=False))
        intersection_edges = inferred_edges_set & gt_edges_set
        union_edges = inferred_edges_set | gt_edges_set
        iou = len(intersection_edges) / len(union_edges)
        return iou
