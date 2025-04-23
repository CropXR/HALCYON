from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ExperimentCondition:
    ko_gene: str | None
    all_genes: tuple[str, ...]
    environment: str | None
    all_environments: tuple[str, ...]

    def as_tensor(self):
        ko_tensor = torch.zeros(len(self.all_genes))
        if self.ko_gene is not None:
            ko_tensor[self.all_genes.index(self.ko_gene)] = 1
        env_tensor = torch.zeros(len(self.all_environments))
        if self.environment is not None:
            env_tensor[self.all_environments.index(self.environment)] = 1
        return torch.cat((ko_tensor, env_tensor), dim=0)
