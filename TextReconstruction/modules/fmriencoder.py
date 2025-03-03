from modules.vit import ViT
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
import torch
from GM.modules.GM import fMRIGM

class fMRIViTEncoderConfig(PretrainedConfig):
    model_type = "fMRIViTEncoder"

    def __init__(
        self,
        fmri_dim, rois_len, embed_dim, depth, num_heads, 
        use_fmrigm = False, fmri2img = False, fmrigm_path = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fmri_dim = fmri_dim
        self.rois_len = rois_len
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.hidden_size = embed_dim
        self.fmri2img = fmri2img
        self.use_fmrigm = use_fmrigm
        self.fmrigm_path = fmrigm_path

class fMRIViTEncoder(PreTrainedModel):
    config_class = fMRIViTEncoderConfig
    def __init__(self, config):
        super(fMRIViTEncoder, self).__init__(config)
        if config.use_fmrigm:
            self.proj = nn.Linear(512, config.embed_dim)
            self.fmrigm = fMRIGM(config.fmri_dim, config.rois_len, 0, 512, 12, 8, None, None)
            self.fmrigm.load_state_dict(torch.load(config.fmrigm_path))
        if config.fmri2img:
            self.proj = nn.Linear(config.fmri_dim, 112 * 112 * 3)
            self.patch_embed = PatchEmbed(112, 16, 3, config.embed_dim)
            self.encoder = ViT(config.embed_dim, 49, config.embed_dim, config.depth, config.num_heads)
        else:
            self.encoder = ViT(config.fmri_dim, config.rois_len, config.embed_dim, config.depth, config.num_heads)
        self.config = config

    def forward(self, encoder_inputs, **kwargs):
        # encoder_inputs shape (batch, roi_num, roi_dim)
        if(self.config.fmri2img):
            encoder_inputs = self.proj(encoder_inputs)
            encoder_inputs = torch.reshape(encoder_inputs, (-1, 3, 112, 112))
            encoder_inputs = self.patch_embed(encoder_inputs)
        elif(self.config.use_fmrigm):
            fmrigm_embedding = self.fmrigm.encode_fmri(encoder_inputs, False)
            fmri_embedding = self.encoder(encoder_inputs)
            encoder_outputs = torch.cat((fmri_embedding, self.proj(fmrigm_embedding)), 1)
        else:
            encoder_outputs = self.encoder(encoder_inputs)
        return encoder_outputs