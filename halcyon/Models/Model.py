from abc import ABC, abstractmethod
from typing import List
from natsort import natsorted
import torch
import torch.nn as nn
from torchdiffeq import odeint
import logging


class Model(ABC, nn.Module):
    def __init__(self, modelled_gene_labels: List[str], device=None):
        super().__init__()
        self.modelled_gene_labels = natsorted(modelled_gene_labels)
        self.gene_index = {gene: i for i, gene in enumerate(self.modelled_gene_labels)}
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = self.get_compatible_device()
        self.reset_conditions()

    @property
    def n_genes(self):
        return len(self.modelled_gene_labels)

    @property
    def n_conditions(self):
        return len(self.env_flags) if hasattr(self, "env_flags") else 0

    @abstractmethod
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Get the derivative at a given time point."""
        pass

    def integrate(
        self,
        initial_state: torch.Tensor,
        t_eval: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the system over time. Provide initial state in form of [Batch, Genes]. Returns tensor with [Batch, Genes, Time]."""
        assert (
            initial_state.shape[1] == self.n_genes
        ), f"Initial state should have shape (B, G) where G={self.n_genes}, but got {initial_state.shape}"
        assert (
            t_eval.dim() == 1
        ), f"t_eval should be a 1D tensor, but got shape {t_eval.shape}"

        output: torch.Tensor = odeint(self, initial_state, t_eval)  # type: ignore
        return output.permute(1, 2, 0)  # Permute dimensions to [Batch, Genes, Time]

    def parse_env_data(
        self, environment_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given the environment data as a tensor, split that into the knockout masks and the environment flags.

        Can be used by children classes to proceed with these KO and env flags.
        Args:
            environment_data (torch.Tensor): tensor of [n_batch, n_genes + n_conditions].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple of knockout masks and environment flags.
        """
        assert (
            environment_data.shape[1] == self.n_genes + self.n_conditions
        ), f"Environment data should be of shape (B, G + C). Expected (B, {self.n_genes + self.n_conditions}) but got {environment_data.shape} where B = {environment_data.shape[0]}, G={self.n_genes}, C={self.n_conditions} and environment_flags = {env_flags}"
        knockout_flags = environment_data[:, : self.n_genes]
        env_flags = environment_data[:, self.n_genes :]
        knockout_masks = 1.0 - knockout_flags
        return knockout_masks, env_flags

    def set_conditions(self, env_data: torch.Tensor):
        """Use the env_data tensor to set the condition (i.e. knockouts and environment flags) for the model.
        On each forward pass the environment flags (1 for KO, 0 for WT) are used as input to the models, and the knockout masks (0 for KO, 1 for WT) are used to mask the output of the models.
        Args:
            env_data (torch.Tensor): Tensor of shape (B, G + C) where B is the batch size, G is the number of genes and C is the number of external conditions.
        """
        # Knockouts should be used as mask after the forwards pass
        ko_masks, env_flags = self.parse_env_data(env_data)
        self.env_flags = env_flags
        # Environments should be appended to the forward pass but not returned (Because they do not change over time)
        self.ko_masks = ko_masks

    def reset_conditions(self):
        self.ko_masks = torch.ones(self.n_genes, device=self.device)  # type: ignore
        self.env_flags = torch.zeros(self.n_conditions, device=self.device)  # type: ignore

    @staticmethod
    def get_compatible_device() -> torch.device:
        """
        Get a compatible device for PyTorch.
        Automatically detects if GPU is compatible and falls back to CPU if not.
        Required because some older GPUs are not supported by recent PyTorch versions.

        Returns:
            torch.device: Compatible device (CPU or GPU)
        """
        if not torch.cuda.is_available():
            logging.info("CUDA not available. Using CPU.")
            return torch.device("cpu")

        try:
            # Get GPU compute capability
            compute_capability = torch.cuda.get_device_capability(0)
            major, minor = compute_capability
            device_name = torch.cuda.get_device_name(0)

            # Get supported architectures from PyTorch
            supported_archs = torch.cuda.get_arch_list()

            # Check if current GPU architecture is supported
            # Architecture names are like 'sm_70', 'sm_75', etc.
            current_arch = f"sm_{major}{minor}"

            if current_arch not in supported_archs:
                logging.warning(
                    f"GPU {device_name} (compute capability {major}.{minor}) is not compatible."
                )
                logging.info(f"Supported architectures: {supported_archs}")
                logging.info("Using CPU instead.")
                return torch.device("cpu")
            else:
                logging.info(
                    f"Using GPU: {device_name} (compute capability {major}.{minor})"
                )
                return torch.device("cuda")

        except RuntimeError as e:
            logging.error(f"Runtime error checking GPU compatibility: {e}")
            logging.info("Falling back to CPU.")
            return torch.device("cpu")
        except AttributeError as e:
            logging.error(f"Attribute error accessing CUDA properties: {e}")
            logging.info("Falling back to CPU.")
            return torch.device("cpu")
        except Exception as e:
            logging.error(f"Unexpected error checking GPU compatibility: {e}")
            logging.info("Falling back to CPU.")
            return torch.device("cpu")
