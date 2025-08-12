import torch
from easydict import EasyDict

from models.backbones.eegmixer import EEGMixer
from models.litmodels.baselitmodellinear import BaseLitModelLinear
from models.litmodels.samlitmodellinear import SAMLitModelLinear
from utils.setup_utils import logger

backbone_dict = {
    "eegmixer": EEGMixer,
}

litmodel_dict = {
    "base": BaseLitModelLinear,
    "sam": SAMLitModelLinear,
}


class ModelMaker:
    def __init__(self, model_name, litmodel_name):

        self.encoder = backbone_dict[model_name]
        self.litmodel = litmodel_dict[litmodel_name]

    def load_model(self, args: EasyDict, **kwargs):
        encoder = self.encoder(args, **kwargs)
        litmodel = self.litmodel(encoder, args)

        return litmodel

    def load_ckpt(self, encoder, path, slack):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        # CKPT load 할 때, 필요한것 맞춰서 구현

        for k in list(state_dict.keys()):
            if k.startswith("model."):
                # if not k.startswith("model.classifier_head"):
                state_dict[k[len("model.") :]] = state_dict[k]

            del state_dict[k]

        msg = encoder.load_state_dict(state_dict, strict=False)
        logger("#manager", msg, slack)

        return encoder
