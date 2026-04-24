import torch.nn as nn


class CoAttentionLayer(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.image_to_text = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_text_1 = nn.LayerNorm(dim)
        self.norm_text_2 = nn.LayerNorm(dim)
        self.norm_image_1 = nn.LayerNorm(dim)
        self.norm_image_2 = nn.LayerNorm(dim)
        self.ffn_text = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ffn_image = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, text_tokens, image_tokens, text_mask=None):
        text_out, _ = self.text_to_image(
            query=text_tokens, key=image_tokens, value=image_tokens
        )
        text_tokens = self.norm_text_1(text_tokens + text_out)
        text_tokens = self.norm_text_2(text_tokens + self.ffn_text(text_tokens))

        image_out, _ = self.image_to_text(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_mask,
        )
        image_tokens = self.norm_image_1(image_tokens + image_out)
        image_tokens = self.norm_image_2(image_tokens + self.ffn_image(image_tokens))

        return text_tokens, image_tokens


class CoAttentionStack(nn.Module):
    def __init__(self, num_layers=2, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CoAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, text_tokens, image_tokens, text_mask=None):
        for layer in self.layers:
            text_tokens, image_tokens = layer(
                text_tokens, image_tokens, text_mask
            )
        return text_tokens, image_tokens
