from dataclasses import dataclass
import torch
from typing_extensions import Self


@dataclass(frozen=True)
class ExperimentCondition:
    """Dataclass to store experimental conditions (both knockout and environment)

    Args:
        ko_gene (tuple[str] | None): Gene that is knocked out. If None, no gene is knocked out. Can also be a tuple of strings for multiple knockouts.
        all_genes (tuple[str, ...]): All genes in the dataset.
        environment (str | None): Environment in which the experiment was conducted. If None, no environment is specified.
        all_environments (tuple[str, ...]): All environments in the dataset.
    """

    ko_gene: tuple[str] | None
    all_genes: tuple[str, ...]
    environment: str | None
    all_environments: tuple[str, ...]

    def as_tensor(self) -> torch.Tensor:
        """Return the experimental condition encoded as tensor.

        First n_genes items represent per gene whether it is knocked out (1) or not (0). The second part of the tensor represents the environment.
        """
        ko_tensor = torch.zeros(len(self.all_genes))
        if self.ko_gene is not None:
            gene_indices = [self.all_genes.index(gene) for gene in self.ko_gene]
            ko_tensor[gene_indices] = 1
        env_tensor = torch.zeros(len(self.all_environments))
        if self.environment is not None:
            env_tensor[self.all_environments.index(self.environment)] = 1
        return torch.cat((ko_tensor, env_tensor), dim=0)

    def return_with_changed_gene_list(self, new_all_genes: list[str]) -> Self:
        """Get ExperimentCondition like current object, but with a different gene (order) list. Used when matching genes in experiment to order how the model expects the genes"""
        return ExperimentCondition(
            ko_gene=self.ko_gene,
            all_genes=tuple(new_all_genes),
            environment=self.environment,
            all_environments=self.all_environments,
        )  # type: ignore
