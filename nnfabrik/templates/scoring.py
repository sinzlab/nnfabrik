import datajoint as dj
import numpy as np
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
from .trained_model import TrainedModelBase


class ScoringBase(dj.Computed):
    """
    Inherit from this class and decorate with your own schema to create a functional
    Score table. This serves as a template for all scores than can be computed with a
    TrainedModel and a dataloader.
    Each master table has an attribute that stores the grand average score across all Neurons.
    The `UnitScore` part table stores the score for all units.
    In order to instantiate a functional table, the default table attriubutes need to be changed.

    This template table is implementing the following logic:
    1) Loading the a trained model from the TrainedModel table. This table needs to have the 'load_model' method.
    2) gettng a dataloader. Minimally, the dataloader returns batches of inputs and targets.
        The dataloader will be built by the Datast table of nnfabrik as default.
        This table needs to have the 'get_dataloader' method
    3) passing the model and the dataloader to a scoring function, defined in the class attribute. The function is expected to
        return either the grand average score or the score per unit.
    4) The average scores and unit scores will be stored in the maser/part tables of this template.

    Attributes:
        trainedmodel_table (Datajoint Table) - an instantiation of the TrainedModelBase
        unit_table (Datajoint Table) - an instantiation of the UnitIDsBase
        scoring_function (function object) - a function that computes the average score (for the master table),
            as well as the unit scores. An example function can be found in this module under 'scoring_function_base'
        scoring_dataset (str) - following nnfabrik convention, this string specifies the key for the 'dataloaders'
            object. The dataloaders object has to contain at least ['train', 'validation', 'test'].
            This string determines, on what data tier the score is computed on. Defaults to the test set.
        scoring_attribute (str) - name of the non-primary attribute of the master and part tables for the score.
        cache (object) - A Store that caches models or datasets, so that they don't need to be recomputed for each
            analysis. Ready to use: an instantiation of the FabrikCache (from nnfabrik.utility.nnf_helper)
    """
    trainedmodel_table = TrainedModelBase
    dataset_table = trainedmodel_table.dataset_table
    function_kwargs = {}
    measure_dataset = "test"
    measure_attribute = "score"
    model_cache = None
    data_cache = None

    @staticmethod
    def measure_function(dataloaders, model, per_unit=True):
        raise NotImplementedError("Scoring Function has to be implemented")

    # table level comment
    table_comment = "A template table for storing results/scores of a TrainedModel"

    @property
    def definition(self):
        definition = """
                # {table_comment}
                -> self.trainedmodel_table
                ---
                {measure_attribute}:      float     # A template for a computed score of a trained model
                {measure_attribute}_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
                """.format(table_comment=self.table_comment, measure_attribute=self.measure_attribute)
        return definition

    class Units(dj.Part):
        @property
        def definition(self):
            definition = """
                # Scores for Individual Neurons
                -> master
                unit_index:                   int
                ---
                unit_{measure_attribute}:     float   # A template for a computed unit score        
                """.format(measure_attribute=self._master.measure_attribute)
            return definition

    def get_model(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        if self.model_cache is None:
            model = self.trainedmodel_table().load_model(key=key,
                                                         include_state_dict=True,
                                                         include_dataloader=False)
        else:
            model = self.model_cache.load(key=key,
                                          include_state_dict=True,
                                          include_dataloader=False)
        return model

    def get_dataloaders(self, key=None):
        if key is None:
            key = self.fetch1('KEY')
        dataloaders = self.dataset_table().get_dataloader(key=key) if self.data_cache is None else self.data_cache.load(key=key)
        return dataloaders[self.measure_dataset]

    def get_overall_score(self, unit_scores):
        return np.mean(unit_scores)

    def insert_unit_scores(self, key, unit_scores):
        key = key.copy()
        for unit_index, unit_score in enumerate(unit_scores):
            key["unit_index"] = unit_index
            key["unit_{}".format(self.measure_attribute)] = unit_score
            self.Units.insert1(key, ignore_extra_fields=True)

    def make(self, key):
        dataloaders = self.get_dataloaders(key=key)
        model = self.get_model(key=key)
        unit_scores = self.measure_function(model=model,
                                                 dataloaders=dataloaders,
                                                 per_unit=True,
                                                 **self.function_kwargs)

        key[self.measure_attribute] = self.get_overall_score(unit_scores)
        self.insert1(key, ignore_extra_fields=True)
        self.insert_unit_scores(key=key, unit_scores=unit_scores)


class SummaryScoringBase(ScoringBase):
    """
    A template scoring table with the same logic as ScoringBase, but for scores that do not have unit scores, but
    an overall score per model only.
    """
    Units = None

    def make(self, key):
        dataloaders = self.get_dataloaders(key=key)
        model = self.get_model(key=key)
        key[self.measure_attribute] = self.measure_function(model=model,
                                                            dataloaders=dataloaders,
                                                            **self.function_kwargs)
        self.insert1(key, ignore_extra_fields=True)


class MeasuresBase(ScoringBase):
    trainedmodel_table = None
    dataset_table = Dataset

    # table level comment
    table_comment = "A template table for storing measures / descriptive statistics of the Dataset"

    @property
    def definition(self):
        definition = """
                    # {table_comment}
                    -> self.dataset_table
                    ---
                    {measure_attribute}:      float     # A template for a computed score of a trained model
                    {measure_attribute}_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
                    """.format(table_comment=self.table_comment, measure_attribute=self.measure_attribute)
        return definition

    class Units(dj.Part):
        @property
        def definition(self):
            definition = """
                # Scores for Individual Neurons
                -> master
                unit_index:                   int     # unit index as extracted by the model
                ---
                unit_{measure_attribute}:     float   # A template for a computed unit score        
                """.format(measure_attribute=self._master.measure_attribute)
            return definition

    def make(self, key):

        dataloaders = self.get_dataloaders(key=key)
        unit_scores = self.measure_function(dataloaders=dataloaders,
                                                   per_unit=True,
                                                   **self.function_kwargs)

        key[self.measure_attribute] = self.get_overall_score(unit_scores)
        self.insert1(key, ignore_extra_fields=True)
        self.insert_unit_scores(key=key, unit_scores=unit_scores)


class SummaryMeasuresBase(MeasuresBase):
    Units = None

    # table level comment
    table_comment = "A template table for storing measures / descriptive statistics of the Dataset"

    def make(self, key):
        dataloaders = self.get_dataloaders(key=key)
        key[self.measure_attribute] = self.measure_function(dataloaders=dataloaders,
                                                            **self.function_kwargs)
        self.insert1(key, ignore_extra_fields=True)