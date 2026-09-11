import torch
import torch.nn as nn
import math

# Example dimensions used in comments:
# batch size=128, sequence length=100, d_model=512, heads=8,
# d_k=64, d_ff=2048, vocabulary size=30000

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # 512
        self.vocab_size = vocab_size #  30,000
        self.embedding = nn.Embedding(vocab_size, d_model) # weight: (30000, 512)

    def forward(self, x):
        # scale because the embedding is not normalized
        return self.embedding(x) * math.sqrt(self.d_model) # (128, 100) -> (128, 100, 512)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding")
            
        self.d_model = d_model # 512
        self.max_seq_len = max_seq_len # 100
        self.dropout = nn.Dropout(dropout) 
        pe = torch.zeros(max_seq_len, d_model) # (100, 512)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # (100, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # (256)

        # in every word, the even indices are sin and the odd indices are cos
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # (100, 256)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # (100, 256)
        pe = pe.unsqueeze(0) # (1, 100, 512)
        # Register buffer to ensure proper tracking of this tensor
        self.register_buffer('pe', pe) # (1, 100, 512)

    def forward(self, x):
        if x.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds max_seq_len {self.max_seq_len}"
            )
        x = x + self.pe[:, :x.size(1), :] # (128, 100, 512) + (1, 100, 512) -> (128, 100, 512)
        return self.dropout(x) # (128, 100, 512)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # (1), broadcast over (128, 100, 512)
        self.bias = nn.Parameter(torch.zeros(1)) # (1), broadcast over (128, 100, 512)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # (128, 100, 1)
        variance = x.var(dim=-1, keepdim=True, unbiased=False) # (128, 100, 1)
        return self.alpha * (x - mean) / (torch.sqrt(variance + self.eps)) + self.bias # (128, 100, 512)

class FeedForward(nn.Module):
    # d_ff is the hidden dimension
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff # 2048
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff) # weight: (2048, 512); (128, 100, 512) -> (128, 100, 2048)
        self.linear_2 = nn.Linear(d_ff, d_model) # weight: (512, 2048); (128, 100, 2048) -> (128, 100, 512)

    def forward(self, x):
        # first linear layer
        hidden_states = self.linear_1(x) # (128, 100, 512) -> (128, 100, 2048)
        hidden_states = torch.relu(hidden_states) # (128, 100, 2048)
        hidden_states = self.dropout(hidden_states) # (128, 100, 2048)

        # second linear layer
        hidden_states = self.linear_2(hidden_states) # (128, 100, 2048) -> (128, 100, 512)
        return hidden_states # (128, 100, 512)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model # 512
        self.n_heads = n_heads # 8
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads # 64
        self.dropout = nn.Dropout(dropout)
        
        self.w_q = nn.Linear(d_model, d_model) # weight: (512, 512);  
        self.w_k = nn.Linear(d_model, d_model) # weight: (512, 512); 
        self.w_v = nn.Linear(d_model, d_model) # weight: (512, 512); 
        self.w_o = nn.Linear(d_model, d_model) # weight: (512, 512); 

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1) # 64
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # (128, 8, 100, 100)
        if mask is not None:
            scores = scores.masked_fill(~mask.to(dtype=torch.bool), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1) # (128, 8, 100, 100)
        attention = self.dropout(attention) # (128, 8, 100, 100)
        return torch.matmul(attention, value) # (128, 8, 100, 64)
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (128, 100, 512)
        key = self.w_k(k) # (128, 100, 512)
        value = self.w_v(v) # (128, 100, 512)
        batch_size = query.size(0)
        
        # split the d_model into n_heads and d_k
        query = query.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (128, 8, 100, 64)
        key = key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (128, 8, 100, 64)
        value = value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (128, 8, 100, 64)
        
        attention = self.attention(query, key, value, mask) # (128, 8, 100, 64)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # (128, 100, 512)
        return self.w_o(attention) # (128, 100, 512)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # (128, 100, 512)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model # 512
        self.n_heads = n_heads # 8
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )
    
    def forward(self, x, mask): # mask is used to mask the padding tokens : no interaction betw padding and other tokens
        x = self.residual_connections[0](
            x, lambda normalized: self.attention(normalized, normalized, normalized, mask)
        )
        x = self.residual_connections[1](x, self.feed_forward)
        return x # (128, 100, 512)
    
class Encoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, n_layers: int):
        super().__init__()
        self.d_model = d_model # 512
        self.n_heads = n_heads # 8
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = LayerNormalization()

    def forward(self, x, mask): # mask is used to mask the padding tokens : no interaction betw padding and other tokens
        for layer in self.layers:
            x = layer(x, mask) # (128, 100, 512)
        return self.norm(x) # (128, 100, 512)

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model # 512
        self.n_heads = n_heads # 8
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout) # self-attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout) # encoder-decoder attention : cross-attention
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda normalized: self.self_attention(
                normalized, normalized, normalized, tgt_mask
            )
        )
        x = self.residual_connections[1](
            x, lambda normalized: self.cross_attention(
                normalized, encoder_output, encoder_output, src_mask
            )
        )
        x = self.residual_connections[2](x, self.feed_forward)
        return x # (128, 100, 512)
    
class Decoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, n_layers: int):
        super().__init__()
        self.d_model = d_model # 512
        self.n_heads = n_heads # 8
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask) # (128, 100, 512)
        return self.norm(x) # (128, 100, 512)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # 512
        self.vocab_size = vocab_size # 30000
        self.proj = nn.Linear(d_model, vocab_size) # weight: (30000, 512); (128, 100, 512) -> (128, 100, 30000)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1) # (128, 100, 30000)
    
class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos_encoder: PositionalEncoding,
        tgt_pos_encoder: PositionalEncoding,
        tgt_output_proj: ProjectionLayer,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos_encoder = src_pos_encoder
        self.tgt_pos_encoder = tgt_pos_encoder
        self.tgt_output_proj = tgt_output_proj
        self.pad_token_id = pad_token_id
    
    def make_src_mask(self, src_seq):
        src_mask = (src_seq != self.pad_token_id) # (128, 100)
        return src_mask.unsqueeze(1).unsqueeze(1) # (128, 1, 1, 100)
    
    def make_tgt_mask(self, tgt_seq):
        seq_len = tgt_seq.size(1)
        padding_mask = (tgt_seq != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=tgt_seq.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(1)
        return padding_mask & causal_mask # (128, 1, 100, 100)

    def forward(self, src_seq, tgt_seq, src_mask=None, tgt_mask=None):
        # 1. prepare the embeddings
        src_embed = self.src_pos_encoder(self.src_embed(src_seq))
        tgt_embed = self.tgt_pos_encoder(self.tgt_embed(tgt_seq))

        # 2. prepare the masks
        if src_mask is None:
            src_mask = self.make_src_mask(src_seq)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt_seq)

        # 3. encode
        encoder_output = self.encoder(src_embed, src_mask) # (128, 100, 512)

        # 4. decode
        decoder_output = self.decoder(tgt_embed, encoder_output, src_mask, tgt_mask) # (128, 100, 512)

        # 5. project the output
        return self.tgt_output_proj(decoder_output) # (128, 100, 30000)

class BuildTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        n_layers: int,
        d_ff: int | None = None,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # 1. create the embedding layers
        self.src_embed = InputEmbeddings(d_model, src_vocab_size) # (128, 100) -> (128, 100, 512)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size) # (128, 100) -> (128, 100, 512)

        # 2. create the positional encoding layers
        self.src_pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout) # (128, 100, 512) -> (128, 100, 512)
        self.tgt_pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout) # (128, 100, 512) -> (128, 100, 512)

        # 3. create the encoder layers
        self.encoder = Encoder(d_model, n_heads, d_ff, dropout, n_layers) # (128, 100, 512)
        self.decoder = Decoder(d_model, n_heads, d_ff, dropout, n_layers) # (128, 100, 512)

        # 4. create the projection layers
        self.tgt_output_proj = ProjectionLayer(d_model, tgt_vocab_size) # (128, 100, 512) -> (128, 100, 30000)

        # 5. create the transformer
        self.transformer = Transformer(
            self.encoder,
            self.decoder,
            self.src_embed,
            self.tgt_embed,
            self.src_pos_encoder,
            self.tgt_pos_encoder,
            self.tgt_output_proj,
            pad_token_id,
        )
        self._initialize_parameters()

    def _initialize_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, src_seq, tgt_seq, src_mask=None, tgt_mask=None):
        return self.transformer(src_seq, tgt_seq, src_mask, tgt_mask)
    
    def make_src_mask(self, src_seq):
        return self.transformer.make_src_mask(src_seq)
    
    def make_tgt_mask(self, tgt_seq):
        return self.transformer.make_tgt_mask(tgt_seq)