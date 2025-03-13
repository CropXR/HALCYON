from mlmmint.Models.Model import Model


class MachineLearningModel(Model):
    def forward(self):
        raise NotImplementedError

    def integrate(self):
        raise NotImplementedError
