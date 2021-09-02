import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LanguageModel(nn.Module):
    '''Language model'''
    def __init__(self, vocab_size=1000, emb_dim=100, hidden_size=100, num_layers=1, bidirectional=False):
        super(LanguageModel, self).__init__()
        self.model_type = 'GRU'
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = emb_dim)
        self.decoder = nn.GRU(
            input_size = emb_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional
        )
        self.bd = bidirectional
        self.hs = hidden_size
        self.vocab_size = vocab_size
        bd_multiplier = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.total_rnns = num_layers
        self.total_rnns *= bd_multiplier
        self.projector = nn.Linear(hidden_size*bd_multiplier, vocab_size)
        self.device = None
    
    def _decode_from_emb(self, x, z, lengths):
        if x.size(-1) == 1:
            x = self.embedding(x).squeeze()
        pack_x = pack_padded_sequence(x, [l+int(self.bd) for l in lengths], enforce_sorted=False)
        packed_output, _ = self.decoder(pack_x, z)
        hidden_states, output_lengths = pad_packed_sequence(packed_output)
        if self.bd:
            forward_states = hidden_states[:-2, :, :self.hs]
            backward_states = hidden_states[2:, :, self.hs:]
            hidden_states = torch.cat([forward_states, backward_states], dim=2)
        projected = self.projector(hidden_states)
        return projected
    
    def decode_cell(self, x, z):
        emb_x = self.embedding(x)[:, :, 0]
        _, state = self.decoder(emb_x, z)
        projected = self.projector(state)
        return state, projected
    
    def forward(self, x, lengths):
        self.device = self.device if self.device is not None else next(self.parameters()).device

        emb_x = self.embedding(x).squeeze(2)
        z = torch.zeros(self.total_rnns, x.size(1), self.hs).to(self.device)
        x_logits = self._decode_from_emb(emb_x, z, lengths)
        return x_logits

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.
    From Pytorch Examples."""

    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, nhead=8, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(emb_dim, nhead, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Embedding(vocab_size, emb_dim)
        self.ninp = emb_dim
        self.decoder = nn.Linear(emb_dim, vocab_size)
        self.vocab_size = vocab_size
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        # nn.init.uniform_(self.decoder.weight, -initrange, initrange)
    
    def decode_cell(self, x, z):
        new_x = torch.cat([z, x], dim=0)
        src = self.encoder(new_x)[:, :, 0] * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return new_x, output[-1:] # no hidden state for transformer

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src)[:, :, 0] * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output[:-1]

class ShiftedConv1d(nn.Module):
    def __init__(self, inc, outc, dilation, **kwargs):
        super(ShiftedConv1d, self).__init__()
        self.dilation = dilation
        self.conv_layer = nn.Sequential(
            nn.Conv1d(inc, outc, 2, padding=dilation, dilation=dilation),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        conv_out = self.conv_layer(x)[:, :, :-self.dilation]
        return self.relu(x[:, :conv_out.size(1)] + conv_out)

class CNNModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, nhead=8, dropout=0.5):
        super(CNNModel, self).__init__()
        self.model_type = 'CNN'
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = emb_dim)
        self.conv_layers = nn.Sequential(
            ShiftedConv1d(emb_dim, hidden_size, 1),
            *[ShiftedConv1d(hidden_size, hidden_size, 2**l_idx)
              for l_idx in range(1, num_layers)]
        )
        self.projector = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
    
    def decode_cell(self, x, z):
        new_x = torch.cat([z, x], dim=0)
        emb_x = self.embedding(new_x).squeeze(2)
        trans_emb_x = emb_x.permute(1, 2, 0)
        out = self.conv_layers(trans_emb_x)
        out = out.permute(2, 0, 1)
        x_logits = self.projector(out)
        return new_x, x_logits[-1:]

    def forward(self, x, lengths):
        emb_x = self.embedding(x).squeeze(2)
        trans_emb_x = emb_x.permute(1, 2, 0)
        out = self.conv_layers(trans_emb_x)
        out = out.permute(2, 0, 1)
        x_logits = self.projector(out)
        return x_logits[:-1]
