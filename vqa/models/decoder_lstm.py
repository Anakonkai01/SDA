import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size=64000, embed_dim=768, hidden_dim=768,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, 8, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_input_ids, memory, memory_key_padding_mask=None):
        embedded = self.dropout(self.embedding(tgt_input_ids))
        lstm_out, _ = self.lstm(embedded)

        attn_out, _ = self.cross_attn(
            query=lstm_out,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )
        out = self.norm(lstm_out + attn_out)
        return self.output_proj(out)
