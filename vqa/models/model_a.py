import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModel, AutoTokenizer

from models.co_attention import CoAttentionStack
from models.decoder_lstm import LSTMDecoder
from models.decoder_transformer import TransformerDecoder


class VQAModelA(nn.Module):
    def __init__(self, decoder_type="lstm", vocab_size=64000,
                 dim=768, clip_dim=512, co_attn_layers=2,
                 co_attn_heads=8, dropout=0.1,
                 lstm_layers=2, transformer_layers=2,
                 transformer_ffn=3072):
        super().__init__()
        self.decoder_type = decoder_type

        self.image_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        self.img_proj = nn.Linear(clip_dim, dim)

        self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.co_attention = CoAttentionStack(
            num_layers=co_attn_layers, dim=dim,
            num_heads=co_attn_heads, dropout=dropout,
        )

        if decoder_type == "lstm":
            self.decoder = LSTMDecoder(
                vocab_size=vocab_size, embed_dim=dim,
                hidden_dim=dim, num_layers=lstm_layers, dropout=dropout,
            )
        else:
            self.decoder = TransformerDecoder(
                vocab_size=vocab_size, d_model=dim, nhead=co_attn_heads,
                num_layers=transformer_layers,
                dim_feedforward=transformer_ffn, dropout=dropout,
            )

    def freeze_encoders(self):
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self, lr_scale=0.1):
        for p in self.image_encoder.parameters():
            p.requires_grad = True
        for p in self.text_encoder.parameters():
            p.requires_grad = True

    def encode(self, pixel_values, input_ids, attention_mask):
        img_out = self.image_encoder(pixel_values).last_hidden_state
        img_tokens = self.img_proj(img_out)

        txt_out = self.text_encoder(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state

        text_mask = attention_mask == 0
        text_enriched, image_enriched = self.co_attention(
            txt_out, img_tokens, text_mask
        )

        memory = torch.cat([text_enriched, image_enriched], dim=1)

        text_len = attention_mask.size(1)
        img_len = img_tokens.size(1)
        memory_mask = torch.cat([
            attention_mask == 0,
            torch.zeros(
                attention_mask.size(0), img_len,
                dtype=torch.bool, device=attention_mask.device,
            ),
        ], dim=1)

        return memory, memory_mask

    def forward(self, pixel_values, input_ids, attention_mask,
                decoder_input_ids, decoder_attention_mask=None):
        memory, memory_mask = self.encode(
            pixel_values, input_ids, attention_mask
        )

        if self.decoder_type == "transformer":
            tgt_len = decoder_input_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=memory.device
            )
            tgt_pad_mask = None
            if decoder_attention_mask is not None:
                tgt_pad_mask = decoder_attention_mask == 0
            logits = self.decoder(
                decoder_input_ids, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=memory_mask,
            )
        else:
            logits = self.decoder(
                decoder_input_ids, memory,
                memory_key_padding_mask=memory_mask,
            )

        return logits

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask,
                 max_length=20, beam_size=1):
        memory, memory_mask = self.encode(
            pixel_values, input_ids, attention_mask
        )
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        generated = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            if self.decoder_type == "transformer":
                tgt_len = generated.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt_len, device=device
                )
                logits = self.decoder(
                    generated, memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_mask,
                )
            else:
                logits = self.decoder(
                    generated, memory,
                    memory_key_padding_mask=memory_mask,
                )

            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token[finished] = eos_id
            generated = torch.cat(
                [generated, next_token.unsqueeze(1)], dim=1
            )
            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        results = []
        for seq in generated:
            tokens = seq[1:].tolist()
            if eos_id in tokens:
                tokens = tokens[:tokens.index(eos_id)]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(text)

        return results
