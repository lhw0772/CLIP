import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


from torch.nn import Dropout
from torch.nn.modules.utils import _pair
import math
from functools import reduce
from operator import mul

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model
class TextEncoder(nn.Module):
    def __init__(self, clip_model, use_forward_text=False):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.use_forward_text = use_forward_text
        self.clip_model = clip_model

    def forward(self, prompts, tokenized_prompts):
        if self.use_forward_text:
            x = self.forward_text(prompts, tokenized_prompts)
        else:
            x = self.forward_default(prompts, tokenized_prompts)

        return x

    def forward_default(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    def forward_text(self, text, tokenized_text):
        x = self.clip_model.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1)] @ self.text_projection

        return x


"""reference: https://github.com/KMnP/vpt/blob/7f3942e49fb062818f17fa11ec4b6d371ef962c8/src/models/vit_prompt/vit.py"""


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_vis_ctx = 16
        n_txt_ctx = 16
        dtype = clip_model.dtype
        txt_ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_ctx_dim = clip_model.visual.conv1.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        #cfg_imsize = cfg.INPUT.SIZE[0]
        cfg_imsize = 224
        patch_size = clip_model.visual.conv1.weight.shape[-1]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        #self.vpt_dropout = Dropout(cfg.TRAINER.DAPT.VIS_DROPOUT)
        self.vpt_dropout = Dropout(0.5)
        vpt_dim = vis_ctx_dim
        clip_patchsize = _pair(patch_size)
        val = math.sqrt(6. / float(3 * reduce(mul, clip_patchsize, 1) + vpt_dim))
        self.vis_ctx = nn.Parameter(torch.zeros(1, n_vis_ctx, vpt_dim, dtype=dtype))  # [1, n_ctx, dim] = [1, 16, 768]
        nn.init.uniform_(self.vis_ctx.data, -val, val)

        print("Initializing a generic context")
        txt_ctx_vectors = torch.empty(n_txt_ctx, txt_ctx_dim, dtype=dtype)
        nn.init.normal_(txt_ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_txt_ctx)
        self.txt_ctx = nn.Parameter(txt_ctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_txt_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        text_prompts = [prompt_prefix + " " + name + "." for name in classnames]  # NOTE: 'X X X X X {cls}'

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in text_prompts])  # NOTE: [cls, 77]

        with torch.no_grad():
            txt_embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", txt_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", txt_embedding[:, 1 + n_txt_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_txt_ctx = n_txt_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def forward_vis(self, x):
        vis_ctx = self.vis_ctx
        B = x.shape[0]
        ctx = self.vpt_dropout(vis_ctx.expand(B, -1, -1)).to(x.dtype)
        prefix = x[:, :1, :]
        suffix = x[:, 1:, :]

        prompt = torch.cat(
            [
                prefix,  # [B, 1, dim]
                ctx,  # [B, n_txt_ctx, dim]
                suffix,  # [B, patches, dim]
            ],
            dim=1,
        )

        return prompt

    def forward_txt(self):
        ctx = self.txt_ctx  # [TXT_NUM_TOKENS, dim] = [16, 512] (default)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_txt_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x, prompts):
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)                       # [B, cls+patches, dim]

        x = prompts(x)

        x = x.permute(1, 0, 2)                          # [B, cls+patches+prompt, dim] NLD -> LND
        x = self.visual.transformer(x)                  # [cls+patches+prompt, B, dim]
        x = x.permute(1, 0, 2)                          # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = 'cuda'

    def forward(self, image):
        visual_prompts = self.prompt_learner.forward_vis
        image_features = self.image_encoder(image.type(self.dtype), visual_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_prompts = self.prompt_learner.forward_txt()
        text_features = self.text_encoder(text_prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features

