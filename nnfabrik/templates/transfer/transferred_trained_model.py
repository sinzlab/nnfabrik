import tempfile
try:
    from collections import Mapping, Sequence
except:
    from collections.abc import Mapping, Sequence

import torch
import os
import datajoint as dj
import numpy as np
from nnfabrik.templates.checkpoint import TrainedModelChkptBase
from nnfabrik.utility.dj_helpers import gitlog, make_hash
from nnfabrik.templates.trained_model import TrainedModelBase


class TransferredTrainedModelBase(TrainedModelBase):
    """
    A modified version of TrainedModel table which enables step-wise table population given the
    step specification. Refer to the corresponding example notebook for a demo.
    """

    table_comment = "Transferred trained models"

    @property
    def definition(self):
        definition = """
        # {table_comment}
        transfer_step:                     int          # transfer step
        -> self().model_table
        -> self().dataset_table
        -> self().trainer_table
        -> self().seed_table
        collapsed_history:                 varchar(64)  # hash of keys from all previous training steps
        data_transfer:                     bool         # flag if we do data transfer 
        ---
        comment='':                        varchar(768) # short description 
        score:                             float        # loss
        output:                            longblob     # trainer object's output
        ->[nullable] self().user_table
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """.format(
            table_comment=self.table_comment
        )
        return definition

    class ModelStorage(TrainedModelBase.ModelStorage):
        pass

    class DataStorage(dj.Part):
        @property
        def definition(self):
            definition = """
            # Contains the data generated by the transfer step, stored externally.
            -> master
            ---
            transfer_data:            attach@{storage}
            """.format(
                storage=self._master.storage
            )
            return definition

    class CollapsedHistory(dj.Part):
        """
        For the result of two or more transfer steps to be uniquely identifiable,
        we compress its entire history (the keys of all previous steps) into a single hash (`collapsed_history`).
        This table keeps track of this process and can be used to recursively retrieve the transfer history.
        """

        definition = """
        next_collapsed_history:                 varchar(64)
        -> master
        """

        @classmethod
        def add_entry(cls, entry):
            entry = {p_key: entry[p_key] for p_key in cls._master.heading.primary_key}
            entry["next_collapsed_history"] = make_hash(entry)
            cls.insert1(entry, skip_duplicates=True)

    def _combine_transfer_recipes(self, transfer_step):
        """
        Combines multiple transfer recipes and their restrictions as specified by post_restr attribute.
        The combination is transfer-step-specific, meaning only the recipes relevant for a specific transfer step would be combined.

        Combining recipes is simple and the user does not need to interact with this method directly. Below is an example:
        Let us assume you have two recipe tables: TrainerRecipe and ModelRecipe, then you can attach all these recipes to your
        TransferTrainedModel table as follows:

        ``` Python
            TransferTrainedModel.transfer_recipe = [TrainerRecipe, ModelRecipe]
        ```

        The rest (combining the recipes and their restrictions) is taken care of by this method.

        Args:
            transfer_step (int): table population transfer step.

        Returns:
            string or datajoint AndList: A single or combined restriction of one or multiple recipes, respectively.
        """

        if not isinstance(self.transfer_recipe, Sequence):
            return self.transfer_recipe
        # else: get the recipes that have an entry for a specific transfer step
        transfer_recipe = []
        for tr in self.transfer_recipe:
            # check if an entry exists for a specific transfer step in the recipe
            if tr & f"transfer_step = {transfer_step}":
                # if it exists add that entry to the list of recipes (relevant for a specific transfer step)
                transfer_recipe.append(tr & f"transfer_step = {transfer_step}")
        if not transfer_recipe:
            return self.proj() - self  # return something empty
        # join all the recipes (and their post_restr)
        joined = transfer_recipe[0]
        if len(transfer_recipe) > 1:
            for t in transfer_recipe[1:]:
                joined *= t  # all combination of recipes
            joined.post_restr = dj.AndList([recipe.post_restr for recipe in self.transfer_recipe])
        return joined

    @property
    def key_source(self):

        # normal entries as a combination of Dataset, Model, Trainer, and Seed tables
        step_0 = self.model_table * self.dataset_table * self.trainer_table * self.seed_table
        # add transfer_step, collapsed_history and data_transfer as prim keys
        key_source = dj.U("transfer_step", "collapsed_history", "data_transfer",) * step_0.proj(
            transfer_step="0",
            collapsed_history='""',
            data_transfer="0",
        )
        if not hasattr(self, "transfer_recipe"):
            return key_source
        # else: expand current entries to follow transfer recipes

        # project (rename) attributes of the existing transferredmodel table to the same name but with prefix "prev"
        prev_transferred_model = self.proj(
            prev_model_fn="model_fn",
            prev_model_hash="model_hash",
            prev_dataset_fn="dataset_fn",
            prev_dataset_hash="dataset_hash",
            prev_trainer_fn="trainer_fn",
            prev_trainer_hash="trainer_hash",
            prev_seed="seed",
            transfer_step="transfer_step + 1",
            prev_collapsed_history="collapsed_history",
            _data_transfer="data_transfer",  # rename so this entry in the recipe is not used for restriction
            prev_step="transfer_step",  # rename so this entry in the recipe is not used for restriction
        ) * dj.U(
            "transfer_step",  # make these attributes primary keys
            "prev_model_fn",
            "prev_model_hash",
            "prev_dataset_fn",
            "prev_dataset_hash",
            "prev_trainer_fn",
            "prev_trainer_hash",
            "prev_seed",
            "prev_collapsed_history",
        )

        # get the current transfer step
        max_transfer_step = prev_transferred_model.fetch("transfer_step").max() if prev_transferred_model else 0

        for transfer_step in range(1, max_transfer_step + 1):
            # get the transfer recipe
            recipe = self._combine_transfer_recipes(transfer_step)
            post_restr = recipe.post_restr if recipe else {}

            # apply the recipe
            transfer_from = prev_transferred_model * recipe
            transfers = (
                self.model_table * self.dataset_table * self.trainer_table * self.seed_table * transfer_from
            )  # combine recipe restriction with all possible training combinations

            transfers = transfers * self.CollapsedHistory().proj(
                prev_collapsed_history="collapsed_history",
                collapsed_history="next_collapsed_history",
                prev_model_fn="model_fn",
                prev_model_hash="model_hash",
                prev_dataset_fn="dataset_fn",
                prev_dataset_hash="dataset_hash",
                prev_trainer_fn="trainer_fn",
                prev_trainer_hash="trainer_hash",
                prev_seed="seed",
                prev_step="transfer_step",
                _data_transfer="data_transfer",
            )  # map previous transferred model to its collapsed history

            transfers = (
                dj.U(
                    "transfer_step",
                    "model_fn",
                    "model_hash",
                    "dataset_fn",
                    "dataset_hash",
                    "trainer_fn",
                    "trainer_hash",
                    "seed",
                    "collapsed_history",
                    "data_transfer",
                )
                & transfers
                & post_restr  # restrict with post_restr
            )
            key_source = key_source.proj() + transfers.proj()
        return key_source

    def get_prev_key(self, key):
        collapsed_history = key["collapsed_history"] if isinstance(key, dict) else key.fetch1("collapsed_history")
        if collapsed_history:
            return self.CollapsedHistory & {"next_collapsed_history": collapsed_history}
        else:  # no history yet (i.e. transfer step 0)
            return self.proj() - self  # return something empty

    def get_full_config(self, key=None, include_state_dict=True, include_trainer=True):
        ret = super().get_full_config(
            key=key,
            include_state_dict=include_state_dict,
            include_trainer=include_trainer,
        )
        prev_key = self.get_prev_key(key)
        # retrieve corresponding model state (and overwrite possibly retrieved state)
        if include_state_dict and (self.ModelStorage & prev_key):
            with tempfile.TemporaryDirectory() as temp_dir:
                state_dict_path = (self.ModelStorage & prev_key).fetch1("model_state", download_path=temp_dir)
                ret["state_dict"] = torch.load(state_dict_path)
                ret["model_config"]["transfer"] = True
        # retrieve data if present (walk backwards in time to find last instance of data transfer)
        while prev_key and key.get("data_transfer"):
            if self.DataStorage & prev_key:
                with tempfile.TemporaryDirectory() as temp_dir:
                    data_path = (self.DataStorage & prev_key).fetch1("transfer_data", download_path=temp_dir)
                    ret["dataset_config"]["transfer_data"] = np.load(data_path)
                break
            prev_key = self.get_prev_key(prev_key)  # go further back
        return ret

    def make(self, key):
        """
        Given key specifying configuration for dataloaders, model and trainer,
        trains the model and saves the trained model.
        """

        fabrikant_name = self.user_table.get_current_user()
        seed = (self.seed_table & key).fetch1("seed")

        dataloaders, model, trainer = self.load_model(key, include_trainer=True, include_state_dict=True, seed=seed)

        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        score, output, model_state = trainer(model=model, dataloaders=dataloaders, seed=seed, uid=key, cb=call_back)
        transfer_data = output.pop("transfer_data", None) if isinstance(output, Mapping) else None

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key)
            key["score"] = score
            key["output"] = output
            key["fabrikant_name"] = fabrikant_name
            comments = []
            comments.append((self.trainer_table & key).fetch1("trainer_comment"))
            comments.append((self.model_table & key).fetch1("model_comment"))
            comments.append((self.dataset_table & key).fetch1("dataset_comment"))
            key["comment"] = self.comment_delimitter.join(comments)

            self.insert1(key)
            self.CollapsedHistory().add_entry(key)

            if key["data_transfer"] and transfer_data:
                data_path = os.path.join(temp_dir, filename + "_transfer_data.npz")
                np.savez(data_path, **transfer_data)
                key["transfer_data"] = data_path
                self.DataStorage.insert1(key, ignore_extra_fields=True)
            filename += ".pth.tar"
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)
            key["model_state"] = filepath
            self.ModelStorage.insert1(key, ignore_extra_fields=True)
