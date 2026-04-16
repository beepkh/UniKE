from ..dataset import CaptionDataset, VQADataset
from ..models.unike import UniKEHyperParams, apply_unike_to_model_mm

try:
    from ..models.mend import MENDHyperParams, MendRewriteExecutor, MendMultimodalRewriteExecutor
    _has_mend = True
except ImportError:
    _has_mend = False

try:
    from ..models.ft import FTHyperParams, apply_ft_to_model, apply_ft_to_multimodal_model
    _has_ft = True
except ImportError:
    _has_ft = False

try:
    from ..models.serac import SERACHparams, SeracRewriteExecutor, SeracMultimodalRewriteExecutor
    _has_serac = True
except ImportError:
    _has_serac = False

try:
    from ..models.ike import IKEHyperParams, apply_ike_to_model, apply_ike_to_multimodal_model, apply_ike_to_per_model
    _has_ike = True
except ImportError:
    _has_ike = False

ALG_DICT = {
    'TPATCHER': apply_unike_to_model_mm,
}
if _has_ft:
    ALG_DICT["FT"] = apply_ft_to_model
if _has_mend:
    ALG_DICT["MEND"] = MendRewriteExecutor().apply_to_model
if _has_serac:
    ALG_DICT["SERAC"] = SeracRewriteExecutor().apply_to_model
if _has_ike:
    ALG_DICT["IKE"] = apply_ike_to_model

ALG_MULTIMODAL_DICT = {
    'TPATCHER': apply_unike_to_model_mm,
    'UNIKE': apply_unike_to_model_mm,
}
if _has_mend:
    ALG_MULTIMODAL_DICT["MEND"] = MendMultimodalRewriteExecutor().apply_to_model
if _has_serac:
    ALG_MULTIMODAL_DICT["SERAC"] = SeracMultimodalRewriteExecutor().apply_to_model
    ALG_MULTIMODAL_DICT["SERAC_MULTI"] = SeracMultimodalRewriteExecutor().apply_to_model
if _has_ike:
    ALG_MULTIMODAL_DICT["IKE"] = apply_ike_to_multimodal_model
if _has_ft:
    ALG_MULTIMODAL_DICT["FT_MULTI"] = apply_ft_to_multimodal_model

PER_ALG_DICT = {}
if _has_ike:
    PER_ALG_DICT["IKE"] = apply_ike_to_per_model

MULTIMODAL_DS_DICT = {
    "caption": CaptionDataset,
    "vqa": VQADataset,
}

