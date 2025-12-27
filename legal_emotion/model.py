import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class LegalEmotionModel(nn.Module):

    def __init__(
        self,
        model_name: str,
        num_emotions: int,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.pool = AttPool(hidden)
        self.lex_proj = nn.Linear(
            num_emotions * 2 + 3,
            hidden,
        )
        self.gate = nn.Sequential(
            nn.Linear(
                hidden * 2,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            hidden,
            num_emotions,
        )
        self.vad_head = nn.Linear(
            hidden,
            3,
        )
        self.count_head = nn.Linear(
            hidden,
            num_emotions,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        lex_counts,
        lex_prior,
        lex_vad,
        token_type_ids=None,
        lex_mask=None,
        lex_vad_mask=None,
    ):
        enc_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if token_type_ids is not None:
            enc_kwargs['token_type_ids'] = token_type_ids
        enc = self.encoder(**enc_kwargs)
        pooled = self.pool(
            enc.last_hidden_state,
            attention_mask,
        )
        if lex_mask is not None:
            mask = lex_mask.to(dtype=lex_counts.dtype)
            if mask.dim() == 1:
                mask = mask.view(
                    -1,
                    1,
                )
            lex_counts = lex_counts * mask
            lex_prior = lex_prior * mask
        if lex_vad_mask is None:
            lex_vad_mask = lex_mask
        if lex_vad_mask is not None:
            mask_v = lex_vad_mask.to(dtype=lex_vad.dtype)
            if mask_v.dim() == 1:
                mask_v = mask_v.view(
                    -1,
                    1,
                )
            lex_vad = lex_vad * mask_v
        lex = torch.cat(
            [lex_counts, lex_prior, lex_vad],
            dim=-1,
        )
        lex = self.lex_proj(lex)
        if lex_mask is not None or lex_vad_mask is not None:
            if lex_mask is None:
                lex_mask = lex_vad_mask
            if lex_vad_mask is None:
                lex_vad_mask = lex_mask
            mask_any = lex_mask
            if (
                lex_mask is not None
                and lex_vad_mask is not None
            ):
                mask_any = torch.maximum(
                    lex_mask.to(dtype=lex.dtype),
                    lex_vad_mask.to(dtype=lex.dtype),
                )
            if mask_any is not None:
                if mask_any.dim() == 1:
                    mask_any = mask_any.view(
                        -1,
                        1,
                    )
                lex = lex * mask_any.to(dtype=lex.dtype)
        gate = self.gate(torch.cat(
                [pooled, lex],
                dim=-1,
            ))
        hidden = self.dropout(pooled * gate + lex * (1 - gate))
        logits = self.classifier(hidden)
        vad = self.vad_head(hidden)
        counts = F.softplus(self.count_head(hidden))
        return (logits, vad, counts)


class AttPool(nn.Module):

    def __init__(
        self,
        hidden,
    ):
        super().__init__()
        self.w = nn.Linear(
            hidden,
            1,
        )

    def forward(
        self,
        x,
        mask,
    ):
        scores = self.w(x).squeeze(-1)
        scores = scores.masked_fill(
            mask == 0,
            -10000.0,
        )
        weights = torch.softmax(
            scores,
            dim=-1,
        ).unsqueeze(-1)
        return torch.sum(
            weights * x,
            dim=1,
        )
