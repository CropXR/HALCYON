from sklearn.metrics import roc_auc_score


from mlmmint.Datasets.ExpressionDataWithConditions import ExpressionDataWithConditions
from mlmmint.Datasets.ExpressionData import ExpressionData
from mlmmint.Models.HybridModel import HybridModel
from mlmmint.Models.MachineLearningModel import MachineLearningModel
from mlmmint.RegulatoryNetwork import RegulatoryNetwork


class Evaluator:
    def __init__(
        self,
        gt_network: RegulatoryNetwork,
        gt_expressions: ExpressionDataWithConditions,
    ):
        self.gt_network = gt_network
        self.gt_expressions = gt_expressions

    def compare_edge_jaccard_gt_vs_inferred_network(
        self, fitted_model: HybridModel | MachineLearningModel
    ) -> float:
        """Compare the jaccard index (intersection over union) of edges in ground truth vs the predicted network.

        Args:
            fitted_model (HybridModel | MachineLearningModel): Model that has been fitted on experimental data

        Returns:
            iou (float): Intersection over union (Jaccard score). 1 is max score, 0 is lowest.
        """
        inferred_network = fitted_model.get_inferred_network()
        inferred_edges_set = set(inferred_network.get_all_interactions(with_data=False))
        gt_edges_set = set(self.gt_network.get_all_interactions(with_data=False))
        intersection_edges = inferred_edges_set & gt_edges_set
        union_edges = inferred_edges_set | gt_edges_set
        iou = len(intersection_edges) / len(union_edges)
        return iou

    def auroc_score_edge_detection(
        self, fitted_model: HybridModel | MachineLearningModel
    ):
        """Calculate area under the receiver operator curve for network inference (edge prediction)

        Args:
            fitted_model (HybridModel | MachineLearningModel): Model that has been fitted and can infer network.
            I.e. it has the .get_inferred_network method that can be called with threshold=None to return raw prediction values.
        """
        inferred_network = fitted_model.get_inferred_network(threshold=None)
        gt_edges_set = set(self.gt_network.get_all_interactions(with_data=False))

        y_true = []
        y_pred = []
        for from_gene, to_gene, edge_dict in inferred_network.get_all_interactions():
            if (from_gene, to_gene) in gt_edges_set:
                y_true.append(1)
            else:
                y_true.append(0)
            y_pred.append(edge_dict["influence"])

        return roc_auc_score(y_true, y_pred)

        # RocCurveDisplay.from_predictions

        # PrecisionRecallDisplay.from_predictions()
