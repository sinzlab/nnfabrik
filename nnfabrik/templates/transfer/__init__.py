import datajoint as dj
import tempfile
import torch
import os
from nnfabrik.main import Model, Dataset, Trainer, Seed, Fabrikant
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
        -> self.model_table
        -> self.dataset_table
        -> self.trainer_table
        -> self.seed_table
        prev_model_fn:                     varchar(64)
        prev_model_hash:                   varchar(64)
        prev_dataset_fn:                   varchar(64)
        prev_dataset_hash:                 varchar(64)
        prev_trainer_fn:                   varchar(64)
        prev_trainer_hash:                 varchar(64)
        collapsed_history:                 varchar(64)
        ---
        comment='':                        varchar(768) # short description 
        score:                             float        # loss
        output:                            longblob     # trainer object's output
        ->[nullable] self.user_table
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        current_model_fn:                  varchar(64)
        current_model_hash:                varchar(64)
        current_dataset_fn:                varchar(64)
        current_dataset_hash:              varchar(64)
        current_trainer_fn:                varchar(64)
        current_trainer_hash:              varchar(64)
        """.format(
            table_comment=self.table_comment
        )
        return definition

    class ModelStorage(TrainedModelBase.ModelStorage):
        pass

    class CollapsedHistory(dj.Part):
        """
        For the result of two or more transfer steps to be uniquely identifiable,
        we compress its entire history (the keys of all previous steps) into a single hash (`collapsed_history`).
        This table keeps track of this process and can be used to recursively retrieve the transfer history.
        """

        definition = """
        collapsed_history:                 varchar(64)
        prev_model_fn:                     varchar(64)
        prev_model_hash:                   varchar(64)
        prev_dataset_fn:                   varchar(64)
        prev_dataset_hash:                 varchar(64)
        prev_trainer_fn:                   varchar(64)
        prev_trainer_hash:                 varchar(64)
        prev_collapsed_history:             varchar(64)
        """

        @classmethod
        def add_entry(cls, key):
            key = {
                "prev_model_fn": key["prev_model_fn"],
                "prev_model_hash": key["prev_model_hash"],
                "prev_dataset_fn": key["prev_dataset_fn"],
                "prev_dataset_hash": key["prev_dataset_hash"],
                "prev_trainer_fn": key["prev_trainer_fn"],
                "prev_trainer_hash": key["prev_trainer_hash"],
                "prev_collapsed_history": key["collapsed_history"],
            }
            key["collapsed_history"] = make_hash(key)
            cls.insert1(key)

    def _transfer_recipe(self, transfer_step):
        """
        Combines multiple transfer recipes and their resitrictions as specified by post_restr attribute.
        The combination is transfer-step-specific, meaning only the recipes relevant for a specific transfer step would be combined.

        Combining recipes are pretty easy and the user does not need to interact with this method directly. Below is an example:
        Let us assume you have two recipe tables: TrainerRecipe and ModelRecipe, the you can attach all these recipes to your
        TransferTrainedModel table as follow:

        ``` Python
            TransferTrainedModel.transfer_recipe = [TrainerRecipe, ModelRecipe]
        ```

        The rest (combining the recipes and their restrictions) is taken care of by this method.

        Args:
            transfer_step (int): table population trasnfer step.

        Returns:
            string or datajoint AndList: A single or combined restriction of one or multiple recipes, respectively.
        """

        if isinstance(self.transfer_recipe, list):
            # get the recipes that have an entry for a specific transfer step
            transfer_recipe = []
            # loop over the transfer recipes
            for tr in self.transfer_recipe:
                # check if an entry exists for a specific transfer step in the recipe
                if tr & f"transfer_step = {transfer_step}":
                    # if it exists add that entry to the list of recipes (relevant for a specific transfer step)
                    transfer_recipe.append(tr & f"transfer_step = {transfer_step}")

            if not transfer_recipe:
                return []
            # join all the recipes (and their post_restr)
            joined = transfer_recipe[0]
            if len(transfer_recipe) > 1:
                for t in transfer_recipe[1:]:
                    joined *= t  # all combination of recipes
                joined.post_restr = dj.AndList(
                    [recipe.post_restr for recipe in self.transfer_recipe]
                )
            return joined
        else:
            return self.transfer_recipe

    @property
    def key_source(self):
        if hasattr(self, "transfer_recipe"):
            # map "prev_"-attributes and "collapsed_history" to their corresponding (updated) collapsed history
            with_collapsed_history = (
                self.proj(
                    "current_model_fn",
                    "current_model_hash",
                    "current_dataset_fn",
                    "current_dataset_hash",
                    "current_trainer_fn",
                    "current_trainer_hash",
                    prev_collapsed_history="collapsed_history",
                )
                * self.CollapsedHistory
            )
            # project (rename) attributes of the existing transferredmodel table to the same name but with prefix "prev"
            prev_transferred_model = with_collapsed_history.proj(
                prev_model_fn="current_model_fn",
                prev_model_hash="current_model_hash",
                prev_dataset_fn="current_dataset_fn",
                prev_dataset_hash="current_dataset_hash",
                prev_trainer_fn="current_trainer_fn",
                prev_trainer_hash="current_trainer_hash",
                prev_step="transfer_step",
                transfer_step="transfer_step + 1",
                collapsed_history="collapsed_history",
            ) * dj.U(
                "transfer_step",  # make these attributes primary keys
                "prev_model_fn",
                "prev_model_hash",
                "prev_dataset_fn",
                "prev_dataset_hash",
                "prev_trainer_fn",
                "prev_trainer_hash",
                "collapsed_history",
            )

            # get the current transfer step
            transfer_step = (
                prev_transferred_model.fetch("transfer_step").max()
                if prev_transferred_model
                else 0
            )

            if transfer_step:

                # get the necessay attributes to filter the prev_transferredmodel with the transfer recipe
                prev_transferred_model = (
                    dj.U(
                        "transfer_step",
                        "prev_model_fn",
                        "prev_model_hash",
                        "prev_dataset_fn",
                        "prev_dataset_hash",
                        "prev_trainer_fn",
                        "prev_trainer_hash",
                        "collapsed_history",
                    )
                    & prev_transferred_model
                )

                # get the entries that match the one in TransferRecipe (all entries that have matching "prev_...")
                transfer_from = prev_transferred_model * self._transfer_recipe(
                    transfer_step
                )

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
                        "prev_model_fn",
                        "prev_model_hash",
                        "prev_dataset_fn",
                        "prev_dataset_hash",
                        "prev_trainer_fn",
                        "prev_trainer_hash",
                        "collapsed_history",
                    )
                    & Model
                    * Dataset
                    * Trainer
                    * Seed
                    * transfer_from  # combine recipe restriction with all possible training combinations
                    & self._transfer_recipe(
                        transfer_step
                    ).post_restr  # restrict with post_rest
                )
                return transfers.proj()

        # normal entries as a combination of Dataset, Model, Trainer, and Seed tables
        step_0 = Model * Dataset * Trainer * Seed

        # add transfer_step and prev_hash as prim keys
        base = dj.U(
            "transfer_step",
            "prev_model_fn",
            "prev_model_hash",
            "prev_dataset_fn",
            "prev_dataset_hash",
            "prev_trainer_fn",
            "prev_trainer_hash",
            "collapsed_history",
        ) * step_0.proj(
            transfer_step="0",
            prev_model_fn='""',
            prev_model_hash='""',
            prev_dataset_fn='""',
            prev_dataset_hash='""',
            prev_trainer_fn='""',
            prev_trainer_hash='""',
            collapsed_history='""',
        )  # train with "prev_"-entries empty
        return base.proj()

    def get_full_config(self, key=None, include_state_dict=True, include_trainer=True):
        ret = super().get_full_config(
            key=key,
            include_state_dict=include_state_dict,
            include_trainer=include_trainer,
        )
        if key["transfer_step"] > 0:
            # retrieve previous key
            prev_prev_key = (
                self.CollapsedHistory & {"collapsed_history": key["collapsed_history"]}
            ).fetch1()
            prev_key = {
                "transfer_step": key["transfer_step"] - 1,
                "model_fn": key["prev_model_fn"],
                "model_hash": key["prev_model_hash"],
                "dataset_fn": key["prev_dataset_fn"],
                "dataset_hash": key["prev_dataset_hash"],
                "trainer_fn": key["prev_trainer_fn"],
                "trainer_hash": key["prev_trainer_hash"],
                "prev_model_fn": prev_prev_key["prev_model_fn"],
                "prev_model_hash": prev_prev_key["prev_model_hash"],
                "prev_dataset_fn": prev_prev_key["prev_dataset_fn"],
                "prev_dataset_hash": prev_prev_key["prev_dataset_hash"],
                "prev_trainer_fn": prev_prev_key["prev_trainer_fn"],
                "prev_trainer_hash": prev_prev_key["prev_trainer_hash"],
                "collapsed_history": prev_prev_key["prev_collapsed_history"],
                "seed": key["seed"],
            }

            # retrieve corresponding model state (and overwrite possibly retrieved state)
            if include_state_dict and (self.ModelStorage & prev_key):
                with tempfile.TemporaryDirectory() as temp_dir:
                    state_dict_path = (self.ModelStorage & prev_key).fetch1(
                        "model_state", download_path=temp_dir
                    )
                    ret["state_dict"] = torch.load(state_dict_path)
        return ret

    def make(self, key):
        """
        Given key specifying configuration for dataloaders, model and trainer,
        trains the model and saves the trained model.
        """

        # lookup the fabrikant corresponding to the current DJ user
        fabrikant_name = Fabrikant.get_current_user()
        seed = (Seed & key).fetch1("seed")

        # load everything
        dataloaders, model, trainer = self.load_model(
            key, include_trainer=True, include_state_dict=True, seed=seed
        )

        # define callback with pinging
        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        # model training
        score, output, model_state = trainer(
            model=model, dataloaders=dataloaders, seed=seed, uid=key, cb=call_back
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key) + ".pth.tar"
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)

            key["score"] = score
            key["output"] = output
            key["fabrikant_name"] = fabrikant_name
            comments = []
            comments.append((self.trainer_table & key).fetch1("trainer_comment"))
            comments.append((self.model_table & key).fetch1("model_comment"))
            comments.append((self.dataset_table & key).fetch1("dataset_comment"))
            key["comment"] = self.comment_delimitter.join(comments)

            key["current_model_fn"], key["current_model_hash"] = (Model & key).fetch1(
                "model_fn", "model_hash"
            )
            key["current_dataset_fn"], key["current_dataset_hash"] = (
                Dataset & key
            ).fetch1("dataset_fn", "dataset_hash")
            key["current_trainer_fn"], key["current_trainer_hash"] = (
                Trainer & key
            ).fetch1("trainer_fn", "trainer_hash")

            self.insert1(key)

            key["model_state"] = filepath

            self.ModelStorage.insert1(key, ignore_extra_fields=True)
            self.CollapsedHistory.add_entry(key)
