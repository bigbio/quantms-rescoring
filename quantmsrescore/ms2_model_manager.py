import pandas as pd
from peptdeep.pretrained_models import ModelManager, model_mgr_settings, \
    psm_sampling_with_important_mods, is_model_zip

# Try to import MODEL_DOWNLOAD_INSTRUCTIONS from peptdeep (version-dependent export)
try:
    from peptdeep.pretrained_models import MODEL_DOWNLOAD_INSTRUCTIONS
except ImportError:
    MODEL_DOWNLOAD_INSTRUCTIONS = (
        "Please download the pretrained models manually from "
        "https://github.com/MannLabs/alphapeptdeep/releases and place them in the model directory."
    )
from peptdeep.model.ms2 import pDeepModel, frag_types, max_frag_charge, ModelMS2Bert, calc_ms2_similarity
from peptdeep.model.rt import AlphaRTModel
from peptdeep.model.ccs import AlphaCCSModel
from peptdeep.model.charge import ChargeModelForModAASeq
import os
from peptdeep.utils import logging
import glob
from alphabase.peptide.fragment import get_charged_frag_types
import torch
import numpy as np
import warnings
from typing import List, Tuple, Optional
import copy
import urllib
import ssl
import certifi
import shutil

def configure_torch_for_hpc(n_threads: int = 1) -> None:
    """
    Configure PyTorch thread settings for HPC environments.

    This function limits PyTorch's internal threading to prevent
    thread explosion when using multiprocessing.

    Parameters
    ----------
    n_threads : int, optional
        Number of threads per PyTorch operation. Default is 1.

    Notes
    -----
    In HPC/Slurm environments, each worker process should use minimal
    internal threads to avoid:
    - Thread competition for CPU cores
    - Excessive memory from thread stacks
    - Performance degradation from context switching
    """
    try:
        # Limit intra-op parallelism (within single operations)
        torch.set_num_threads(n_threads)
        # Limit inter-op parallelism (between independent operations)
        torch.set_num_interop_threads(n_threads)
    except RuntimeError:
        # Threads already configured (can only be set once per process)
        pass


# Opt-in PyTorch thread configuration via environment variable
# Set QUANTMS_HPC_MODE=1 to enable automatic thread limiting at import time
# For explicit control, call configure_torch_for_hpc() directly in your code
if os.environ.get("QUANTMS_HPC_MODE", "").lower() in ("1", "true", "yes"):
    configure_torch_for_hpc(n_threads=1)


class MS2ModelManager(ModelManager):
    """Extended ModelManager that uses MS2pDeepModel for MS2 predictions.

    Note: This class intentionally does not call super().__init__() because it
    replaces the ms2_model with MS2pDeepModel. Calling the parent would create
    duplicate model instances.
    """

    def __init__(self,
                 mask_modloss: bool = False,
                 device: str = "gpu",
                 model_dir: str = ".",
                 ):
        # Initialize attributes expected by parent class methods
        self._train_psm_logging = True

        self.ms2_model: pDeepModel = MS2pDeepModel(
            mask_modloss=mask_modloss, device=device
        )
        self.rt_model: AlphaRTModel = AlphaRTModel(device=device)
        self.ccs_model: AlphaCCSModel = AlphaCCSModel(device=device)

        self.charge_model: ChargeModelForModAASeq = ChargeModelForModAASeq(
            device=device
        )
        self.model_url = "https://github.com/MannLabs/alphapeptdeep/releases/download/pre-trained-models/pretrained_models_v3.zip"

        ms2_model_files = glob.glob(os.path.join(model_dir, "*ms2.pth"))
        if ms2_model_files:
            self.load_external_models(ms2_model_file=ms2_model_files[0])
            self.model_str = model_dir
        else:
            self.download_model_path = os.path.join(model_dir, "pretrained_models_v3.zip")
            self._download_models(self.download_model_path)
            self.load_installed_models(self.download_model_path)
            self.model_str = "generic"
        self.pretrained_ms2_model = copy.deepcopy(self.ms2_model)
        self.reset_by_global_settings(reload_models=False)

    def __str__(self):
        return self.model_str

    def _download_models(self, model_zip_file_path: str, skip_if_exists: bool = True) -> None:
        """
        Download models if not done yet.

        Uses streaming download to avoid loading entire file into memory,
        and a longer timeout (300s) for large files on slow connections.

        Parameters
        ----------
        model_zip_file_path : str
            Path where the model zip file will be saved.
        skip_if_exists : bool, optional
            If True (default), skip download when file already exists.
            If False, raise FileExistsError when file exists.
        """
        url = self.model_url
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Disallowed URL scheme: {parsed.scheme}")

        if os.path.exists(model_zip_file_path):
            if not skip_if_exists:
                raise FileExistsError(f"Model file already exists: {model_zip_file_path}")
            logging.debug(f"Model file already exists, skipping download: {model_zip_file_path}")
        else:
            logging.info(f"Downloading pretrained models from {url} to {model_zip_file_path} ...")
            try:
                parent_dir = os.path.dirname(model_zip_file_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                context = ssl.create_default_context(cafile=certifi.where())
                # Use streaming download with longer timeout for large model files
                # timeout=300s (5 min) for slow connections; stream in 1MB chunks
                with urllib.request.urlopen(url, context=context, timeout=300) as response:  # nosec B310
                    with open(model_zip_file_path, "wb") as out_file:
                        shutil.copyfileobj(response, out_file, length=1024 * 1024)  # 1MB chunks
            except Exception as e:
                # Clean up partial download on failure
                if os.path.exists(model_zip_file_path):
                    try:
                        os.remove(model_zip_file_path)
                    except OSError:
                        pass  # Best-effort cleanup; file may already be removed or locked
                raise FileNotFoundError(
                    f"Downloading model failed: {e}.\n" + MODEL_DOWNLOAD_INSTRUCTIONS
                ) from e

            logging.info("Successfully downloaded pretrained models.")
        if not is_model_zip(model_zip_file_path):
            # Clean up invalid/corrupted file
            try:
                os.remove(model_zip_file_path)
            except OSError:
                pass  # Best-effort cleanup; file may already be removed or locked
            raise ValueError(
                f"Local model file is not a valid zip: {model_zip_file_path}.\n"
                f"The invalid file has been removed. Please try again.\n"
                f"Or: {MODEL_DOWNLOAD_INSTRUCTIONS}"
            )

    def ms2_fine_tuning(self, psms_df: pd.DataFrame,
                        match_intensity_df: pd.DataFrame,
                        psm_num_to_train_ms2: int = 100000000,
                        use_grid_nce_search: bool = False,
                        top_n_mods_to_train: int = 10,
                        psm_num_per_mod_to_train_ms2: int = 50,
                        psm_num_to_test_ms2: int = 0,
                        epoch_to_train_ms2: int = 20,
                        train_verbose: bool = False,
                        force_transfer_learning: bool = False):

        self.psm_num_to_train_ms2 = psm_num_to_train_ms2
        self.use_grid_nce_search = use_grid_nce_search
        self.top_n_mods_to_train = top_n_mods_to_train
        self.psm_num_per_mod_to_train_ms2 = psm_num_per_mod_to_train_ms2
        self.psm_num_to_test_ms2 = psm_num_to_test_ms2
        self.train_verbose = train_verbose
        self.force_transfer_learning = force_transfer_learning
        self.epoch_to_train_ms2 = epoch_to_train_ms2
        self.train_ms2_model(psms_df, match_intensity_df)

    def load_installed_models(self, download_model_path: str = "pretrained_models_v3.zip"):
        """Load built-in MS2/CCS/RT models.

        Parameters
        ----------
        download_model_path : str, optional
            The path of model zip file.
            Defaults to 'pretrained_models_v3.zip'.
        """

        self.ms2_model.load(
            download_model_path, model_path_in_zip="generic/ms2.pth"
        )
        self.rt_model.load(download_model_path, model_path_in_zip="generic/rt.pth")
        self.ccs_model.load(
            download_model_path, model_path_in_zip="generic/ccs.pth"
        )
        self.charge_model.load(
            download_model_path, model_path_in_zip="generic/charge.pth"
        )

    def _build_intensity_df(self, matched_intensity_df: pd.DataFrame) -> pd.DataFrame:
        """Build intensity DataFrame with all charged fragment types.

        Parameters
        ----------
        matched_intensity_df : pd.DataFrame
            The matched fragment intensities.

        Returns
        -------
        pd.DataFrame
            Intensity DataFrame with columns for all charged fragment types,
            filled with 0.0 for missing columns.
        """
        inten_df = pd.DataFrame()
        for frag_type in self.ms2_model.charged_frag_types:
            if frag_type in matched_intensity_df.columns:
                inten_df[frag_type] = matched_intensity_df[frag_type]
            else:
                inten_df[frag_type] = 0.0
        return inten_df

    def train_ms2_model(
            self,
            psm_df: pd.DataFrame,
            matched_intensity_df: pd.DataFrame,
    ):
        """
        Using matched_intensity_df to train/fine-tune the ms2 model.

        1. It will sample `n=self.psm_num_to_train_ms2` PSMs into training dataframe (`tr_df`) for fine-tuning.
        2. This method will also consider some important PTMs (`n=self.top_n_mods_to_train`) into `tr_df` for fine-tuning.
        3. If `self.use_grid_nce_search==True`, this method will call `self.ms2_model.grid_nce_search` to find the best NCE and instrument.

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM dataframe for fine-tuning

        matched_intensity_df : pd.DataFrame
            The matched fragment intensities for `psm_df`.
        """
        if self.psm_num_to_train_ms2 > 0:
            if self.psm_num_to_train_ms2 < len(psm_df):
                tr_df = psm_sampling_with_important_mods(
                    psm_df,
                    self.psm_num_to_train_ms2,
                    self.top_n_mods_to_train,
                    self.psm_num_per_mod_to_train_ms2,
                ).copy()
            else:
                tr_df = psm_df
            if len(tr_df) > 0:
                tr_inten_df = self._build_intensity_df(matched_intensity_df)

                if self.use_grid_nce_search:
                    self.nce, self.instrument = self.ms2_model.grid_nce_search(
                        tr_df,
                        tr_inten_df,
                        nce_first=model_mgr_settings["transfer"]["grid_nce_first"],
                        nce_last=model_mgr_settings["transfer"]["grid_nce_last"],
                        nce_step=model_mgr_settings["transfer"]["grid_nce_step"],
                        search_instruments=model_mgr_settings["transfer"][
                            "grid_instrument"
                        ],
                    )
                    tr_df["nce"] = self.nce
                    tr_df["instrument"] = self.instrument
                else:
                    self.set_default_nce_instrument(tr_df)
        else:
            tr_df = pd.DataFrame()

        if self.psm_num_to_test_ms2 > 0:
            if len(tr_df) > 0:
                test_psm_df = psm_df[~psm_df.sequence.isin(set(tr_df.sequence))].copy()
                if len(test_psm_df) > self.psm_num_to_test_ms2:
                    test_psm_df = test_psm_df.sample(n=self.psm_num_to_test_ms2)
                elif len(test_psm_df) == 0:
                    logging.info(
                        "No enough PSMs for testing MS2 models, "
                        "please reduce the `psm_num_to_train_ms2` "
                        "value according to overall PSM numbers. "
                    )
                    test_psm_df = pd.DataFrame()
            else:
                test_psm_df = psm_df.copy()
                tr_inten_df = self._build_intensity_df(matched_intensity_df)
            self.set_default_nce_instrument(test_psm_df)
        else:
            test_psm_df = pd.DataFrame()

        if len(test_psm_df) > 0:
            logging.info("Testing pretrained MS2 model on testing df:")
            pretrained, metrics_desc = self.ms2_model.test(test_psm_df, tr_inten_df)
            logging.info("\n" + str(metrics_desc))

        if len(tr_df) > 0:
            if self._train_psm_logging:
                logging.info(
                    f"{len(tr_df)} PSMs for MS2 model training/transfer learning"
                )
            self.ms2_model.train(
                tr_df,
                fragment_intensity_df=tr_inten_df,
                batch_size=self.batch_size_to_train_ms2,
                epoch=self.epoch_to_train_ms2,
                warmup_epoch=self.warmup_epoch_to_train_ms2,
                lr=self.lr_to_train_ms2,
                verbose=self.train_verbose,
            )
            logging.info(
                "Testing refined MS2 model on training df:\n"
                + str(self.ms2_model.test(tr_df, tr_inten_df)[-1])
            )
        if len(test_psm_df) > 0:
            logging.info("Testing refined MS2 model on testing df:")
            fine_tuning, metrics_desc = self.ms2_model.test(test_psm_df, tr_inten_df)
            logging.info("\n" + str(metrics_desc))

        if len(test_psm_df) > 0:
            if pretrained["SA"].median() > fine_tuning["SA"].median():
                logging.info("fine_tuning model is not better than pretrained for test dataset")
                if self.force_transfer_learning:
                    logging.info("Forced save fine-tune model")
                    self.model_str = "retrained_model"
                else:
                    logging.info("Save original model")
                    self.ms2_model = self.pretrained_ms2_model
            else:
                self.model_str = "retrained_model"

        else:
            self.model_str = "retrained_model"

    def save_ms2_model(self, save_model_dir):
        self.ms2_model.save(os.path.join(save_model_dir, "retained_ms2.pth"))


class MS2pDeepModel(pDeepModel):
    """
    `ModelInterface` for MS2 prediction models

    Parameters
    ----------
    charged_frag_types : List[str]
        Charged fragment types to predict
    dropout : float, optional
        Dropout rate, by default 0.1
    model_class : torch.nn.Module, optional
        MS2 model class (should be a subclass of torch.nn.Module implementing the MS2 prediction interface), by default ModelMS2Bert
    device : str, optional
        Device to run the model, by default "gpu"
    override_from_weights : bool, optional default False
        Override the requested charged frag types from the model weights on loading.
        This allows to predict all fragment types supported by the weights even if the user doesn't know what fragments types are supported by the weights.
        Thereby, the model will always be in a safe to predict state.
    mask_modloss : bool, optional (deprecated)
        Mask the modloss fragments, this is deprecated and will be removed in the future. To mask the modloss fragments,
        the charged_frag_types should not include the modloss fragments.

    """

    def __init__(
            self,
            charged_frag_types=None,
            dropout=0.1,
            model_class: torch.nn.Module = ModelMS2Bert,
            device: str = "gpu",
            mask_modloss: Optional[bool] = None,
            override_from_weights: bool = False,
            **kwargs,  # model params
    ):
        # Avoid function call in default argument (Ruff B008)
        # Evaluated once at definition time, not at each call
        if charged_frag_types is None:
            charged_frag_types = get_charged_frag_types(frag_types, max_frag_charge)

        super().__init__(
            charged_frag_types=charged_frag_types,
            dropout=dropout,
            model_class=model_class,
            device=device,
            mask_modloss=mask_modloss,
            override_from_weights=override_from_weights,
            **kwargs,  # model params
        )
        if mask_modloss is not None:
            warnings.warn(
                "mask_modloss is deprecated and will be removed in the future. To mask the modloss fragments, "
                "the charged_frag_types should not include the modloss fragments."
            )

    def _set_batch_predict_data(
            self,
            batch_df: pd.DataFrame,
            predicts: np.ndarray,
            **_kwargs,
    ):
        apex_intens = predicts.reshape((len(batch_df), -1)).max(axis=1)
        apex_intens[apex_intens <= 0] = 1
        predicts /= apex_intens.reshape((-1, 1, 1))
        predicts[predicts < self.min_inten] = 0.0
        # mask out predicted charged frag types that are not in the requested charged_frag_types
        columns_mask = np.isin(
            self.model.supported_charged_frag_types, self.charged_frag_types
        )
        predicts = predicts[:, :, columns_mask]

        if self._predict_in_order:
            self.predict_df.values[
            batch_df.frag_start_idx.values[0]: batch_df.frag_stop_idx.values[-1], :
            ] = predicts.reshape((-1, len(self.charged_frag_types)))
        else:
            update_sliced_fragment_dataframe(
                self.predict_df,
                self.predict_df.to_numpy(copy=True),
                predicts.reshape((-1, len(self.charged_frag_types))),
                batch_df[["frag_start_idx", "frag_stop_idx"]].values,
            )

    def test(
            self,
            precursor_df: pd.DataFrame,
            fragment_intensity_df: pd.DataFrame,
            default_instrument: str = "Lumos",
            default_nce: float = 30.0,
    ) -> pd.DataFrame:
        if "instrument" not in precursor_df.columns:
            precursor_df["instrument"] = default_instrument
        if "nce" not in precursor_df.columns:
            precursor_df["nce"] = default_nce
        columns = np.intersect1d(
            self.charged_frag_types,
            fragment_intensity_df.columns.values,
        )
        return calc_ms2_similarity(
            precursor_df,
            self.predict(precursor_df, reference_frag_df=fragment_intensity_df)[
                columns
            ],
            fragment_intensity_df=fragment_intensity_df[columns],
        )


def update_sliced_fragment_dataframe(
        fragment_df: pd.DataFrame,
        fragment_df_vals: np.ndarray,
        values: np.ndarray,
        frag_start_end_list: List[Tuple[int, int]],
        charged_frag_types: List[str] = None,
):
    """
    Set the values of the slices `frag_start_end_list=[(start,end),(start,end),...]`
    of fragment_df.

    Parameters
    ----------
    fragment_df : pd.DataFrame
        fragment dataframe to set the values

    fragment_df_vals : np.ndarray
        The `fragment_df.to_numpy(copy=True)` to prevent readonly assignment.

    values : np.ndarray
        values to set

    frag_start_end_list : List[Tuple[int,int]]
        e.g. `[(start,end),(start,end),...]`

    charged_frag_types : List[str], optional
        e.g. `['b_z1','b_z2','y_z1','y_z2']`.
        If None, the columns of values should be the same as fragment_df's columns.
        It is much faster if charged_frag_types is None as we use numpy slicing,
        otherwise we use pd.loc (much slower).
        Defaults to None.
    """
    frag_slice_list = [slice(start, end) for start, end in frag_start_end_list]
    frag_slices = np.r_[tuple(frag_slice_list)]
    if charged_frag_types is None or len(charged_frag_types) == 0:
        fragment_df_vals[frag_slices, :] = values.astype(fragment_df_vals.dtype)
        fragment_df.iloc[frag_slices, :] = values.astype(fragment_df_vals.dtype)
    else:
        charged_frag_idxes = [
            fragment_df.columns.get_loc(c) for c in charged_frag_types
        ]
        fragment_df.iloc[frag_slices, charged_frag_idxes] = values.astype(
            fragment_df_vals.dtype
        )
        fragment_df_vals[frag_slices] = fragment_df.values[frag_slices]
