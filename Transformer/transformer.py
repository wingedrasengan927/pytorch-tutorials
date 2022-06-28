import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class EncoderDecoder(nn.Module):
    '''
    A standard encoder-decoder architecture.
    '''

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        '''
        Process masked source and target sequences
        '''
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    '''
    Define Standard linear + softmax generation step
    '''

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def make_clones(module, N):
    '''
    Produce N identical layers
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    '''
    The Encoder is a stack of N Encoder Blocks each with their own weights.
    '''

    def __init__(self, encoder_block, N):
        super(Encoder, self).__init__()
        self.blocks = make_clones(encoder_block, N)
        self.norm = nn.LayerNorm(encoder_block.d_model)

    def forward(self, x, mask):
        '''
        Pass the input (and mask) through each layer in turn
        '''
        for block in self.blocks:
            x = block(x, mask)
        # normalize the final output
        return self.norm(x)

class SublayerConnection(nn.Module):
    '''
    A skip layer connection. It takes the layer and the input to the layer
    and performs the skip layer operation: x + layer(x).
    In addition, the input is normalized before feeding into the layer and
    dropout is applied to the output of the layer.

    Parameters
    -----------
    inpt_size - int
        The dimension of the input tensor.
    p_dropout - float
        The dropout probability to be applied to the output of the layer
    '''

    def __init__(self, inpt_size, p_dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(inpt_size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, sublayer):
        '''
        Apply residual connection to any sublayer of the same size
        '''
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    '''
    An Encoder Block consists of a multi-head self-attention layer followed
    by a feed-forward network with skip connections for each layer.

    Parameters
    ----------
    d_model - int
        Size of the input dimension which is usually the embedding size
    self_attn - function which calls the module
        The multi-head self-attention layer
    feed_forward - nn.Module
        The feed-forward network
    p_dropout - float
        The dropout probability to be applied to the outputs of each layer
    '''

    def __init__(self, d_model, self_attn, feed_forward, p_dropout):
        super(EncoderBlock, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = make_clones(SublayerConnection(d_model, p_dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 
        return self.sublayer[1](x, self.feed_forward) 

class Decoder(nn.Module):
    '''
    The Decoder is a stack of N Decoder Blocks each with their own weights.
    '''
    
    def __init__(self, block, N):
        super(Decoder, self).__init__()
        self.blocks = make_clones(block, N)
        self.norm = nn.LayerNorm(block.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for block in self.blocks:
            x = block(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    '''
    The Deocder Block is composed of a multi-head self attention layer followed by a 
    multi-head encoder attention layer followed by a feed forward network with skip
    connections for each layer.
    '''

    def __init__(self, d_model, self_attn, src_attn, feed_forward, p_dropout):
        super(DecoderBlock, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = make_clones(SublayerConnection(d_model, p_dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        x - Floating point tensor of size (batch_size, tgt_sentence_length, embedding_dim)
            The target sequence tensor which is input to the decoder
        memory - Floating point tensor of size (batch_size, inpt_sentence_length, embedding_dim)
            The final output of the encoder.
        src_mask - Arbitary Tensor of shape (*, 1, inpt_sentence_length)
            mask to be applied on encoder-attention scores to hide certain words in the input sentence.
        tgt_mask - Arbitary Tensor of shape (*, tgt_sentence_length, tgt_sentence_length)
            Mask to be applied on self-attention scores in the decoder to prevent the current word from 
            looking at subsequent future words.
        '''
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    '''
    A mask that prevents current word from looking at subsequent future words.
    Usually applied to the decoder input.
    '''
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0

def attention(query, key, value, mask=None, dropout=None):
    '''
    Compute scaled dot product attention.

    Parameters
    -----------
    query, key, value - Floating point tensors of shape (batch_size, n_heads, sentence_length, head_dim)
        The projected multi-head query, key, and value tensors.
        batch_size, no. of heads, and head dim should be the same for all of them.
        However, queries and keys can have different lengths, while keys and values
        should have the same length.
    mask - Tensor of arbitary size (*, query_length, key_length) or (*, key_length)
        The mask will be broadcasted and applied across the raw attention scores computed from queries and keys.

    Returns
    --------
    attn weighed values - Floating point tensors of shape (batch_size, n_heads, query_length, head_dim)
        The attention weighed value tensors
    attn -  Floating point tensors of shape (batch_size, n_heads, query_length, key_length)
        The attn probability scores computed for each head for a batch of queries and keys
    '''
    d_k = query.size(-1)
    # (batch, heads, q_size, d_k) x (batch, heads, d_k, k_size) = (batch, heads, q_size, k_size)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # (batch, heads, q_size, k_size) x (batch, heads, k_size, d_k) = (batch, heads, q_size, d_k)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    '''
    Compute Multi-head self-attention

    Parameters
    -----------
    n_heads - int
        Number of heads the query, key, and value tensors should be split into
    d_model - int
        Size of the input dimension which is usually the embedding size
    p_dropout - float
        Dropout probability to be applied to the attention scores
    '''
    
    def __init__(self, n_heads, d_model, p_dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads  == 0
        # we assume d_v always equals d_k
        self.d_k = d_model // n_heads # dim of each head
        self.n_heads = n_heads
        # the first three layers project the input into queries, keys, and values.
        # the last layer performs a 1x1 convolution operation on the final concatenated attention vector.
        self.linear_layers = make_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None # placeholder for attention scores
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            # broadcast the same mask to all the heads
            mask = mask.unsqueeze(1)

        batch_size = query.size(0) 

        # 1) project the input into queries, keys, and values
        # and split each of them into multiple heads: d_model => n_heads x d_k
        query, key, value = [
            lin_layer(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) 
            for lin_layer, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply self-attention on all the projected multi-head tensors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concat all the heads using a view
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_k)
        )

        del query
        del key
        del value

        # apply a final linear layer (1 x 1 convolution)
        return self.linear_layers[-1](x)

class PositionwiseFeedForward(nn.Module):
    '''
    A simple feed-forward network applied on the attention tensor at the end

    Parameters
    -----------
    d_model - int
        Size of the input dimension which is usually the embedding size
    dff - int
        size of hidden state in the network
    p_dropout - float
        dropout probability to be applied to the hidden layer
    '''
    def __init__(self, d_model, d_ff, p_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        out = F.relu(self.w_1(x))
        out = self.dropout(out)
        out = self.w_2(out)
        return out

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, p_dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p_dropout)

        # compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)