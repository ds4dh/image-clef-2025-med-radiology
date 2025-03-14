import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Tuple

from math import log2, comb, gcd, prod


Code = namedtuple("Code", ["index", "value"])


class DownConvBlockInternal(nn.Module):

    def __init__(self, c_i, c_o, r):
        super(DownConvBlockInternal, self).__init__()

        self.c_i = c_i
        self.c_o = c_o
        self.r = r

        self.res_conv = nn.Conv2d(self.c_i, self.c_o, 1, padding=0, stride=self.r)
        self.seq = nn.Sequential(
            nn.Conv2d(self.c_i, self.c_o, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.c_o, self.c_o, kernel_size=5, padding=2, stride=self.r, groups=1),
            nn.BatchNorm2d(self.c_o),
            # nn.InstanceNorm2d(self.c_o),
        )

    def forward(self, inp):
        res = self.res_conv(inp)
        inp = self.seq(inp)
        inp = inp + res
        inp = inp.relu_()
        inp = F.leaky_relu(inp)

        return inp


class DownConvBlockExternal(nn.Module):

    def __init__(self, c_i, c_o, r):
        super(DownConvBlockExternal, self).__init__()

        self.c_i = c_i
        self.c_o = c_o
        self.r = r

        self.res_conv = nn.Conv2d(self.c_i, self.c_o, kernel_size=1, padding=0, stride=self.r)
        self.seq = nn.Sequential(
            nn.Conv2d(self.c_i, self.c_i, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.c_i, self.c_i, kernel_size=5, padding=2, stride=1),
            nn.Conv2d(self.c_i, self.c_o, kernel_size=1, padding=0, stride=self.r),
            nn.BatchNorm2d(self.c_o)
            # nn.InstanceNorm2d(self.c_o)
        )

    def forward(self, inp):
        res = self.res_conv(inp)
        inp = self.seq(inp)
        inp = inp + res
        inp = F.leaky_relu(inp)

        return inp


class ConvEmbedding(nn.Module):

    def __init__(self, init_dict):
        super(ConvEmbedding, self).__init__()

        self.r_list = init_dict["sampling_ratio_list"]
        self.c_list = init_dict["channels_list"]
        self.c_o = init_dict["num_out_channels"]

        self.dropout = init_dict["dropout"]
        self.num_blocks = len(self.r_list)

        assert len(self.c_list) == 1 + len(self.r_list)

        self.down_blocks_int = nn.ModuleList()
        self.down_blocks_ext = nn.ModuleList()
        for i_b in range(self.num_blocks):
            self.down_blocks_int.append(
                DownConvBlockInternal(self.c_list[i_b], self.c_list[i_b + 1], self.r_list[i_b]))
            self.down_blocks_ext.append(
                DownConvBlockExternal(self.c_list[i_b], self.c_list[-1], prod(self.r_list[i_b:])))

        self.conv_channel = nn.Conv2d(self.c_list[-1] * (self.num_blocks + 1), self.c_o, kernel_size=1)

    def forward(self, inp):

        b, c, h, w = inp.shape
        for i_b in range(self.num_blocks):
            if i_b == 0:
                res = self.down_blocks_ext[0](inp)
            else:
                res = torch.cat((res, self.down_blocks_ext[i_b](inp)), 1)
            inp = self.down_blocks_int[i_b](inp)

        res = torch.cat((res, inp), 1)
        out = self.conv_channel(res)

        return out


class TransformerSeqGen(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_dim = config["input_dim"]
        hidden_dim = config["hidden_dim"]
        vocab_size = config["vocab_size"]
        max_len = config["max_len"]
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size

    def forward(self, input_vector, target_seq):
        """
        input_vector: [batch, input_dim]
        target_seq: [batch, tgt_len] (contains token indices)
        """
        batch_size, tgt_len = target_seq.shape

        enc_input = self.input_projection(input_vector).unsqueeze(0)  # [1, batch, hidden_dim]

        tgt_emb = self.token_embedding(target_seq)  # [batch, tgt_len, hidden_dim]

        pos_ids = torch.arange(tgt_len, device=target_seq.device).unsqueeze(0).expand(batch_size, -1)
        tgt_emb = tgt_emb + self.pos_embedding(pos_ids)

        tgt_emb = tgt_emb.permute(1, 0, 2)  # [seq_len, batch, hidden_dim]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(target_seq.device)

        output = self.transformer(
            enc_input, tgt_emb, tgt_mask=tgt_mask
        )  # [tgt_len, batch, hidden_dim]

        output = output.permute(1, 0, 2)  # [batch, tgt_len, hidden_dim]
        logits = self.output_layer(output)  # [batch, tgt_len, vocab_size]

        return logits

    def generate(self, input_vector, eos_token, max_length=50):

        input_vector = self.input_projection(input_vector).unsqueeze(0).permute(1, 0, 2)  # [1, batch, hidden_dim]
        generated_seq = [torch.tensor([0], device=input_vector.device)]  # Start token

        for _ in range(max_length):
            tgt_seq = torch.cat(generated_seq, dim=0).unsqueeze(0)  # [1, seq_len]
            tgt_emb = self.token_embedding(tgt_seq)

            pos_ids = torch.arange(tgt_seq.shape[1], device=tgt_seq.device).unsqueeze(0)
            tgt_emb = tgt_emb + self.pos_embedding(pos_ids)

            output = self.transformer(
                input_vector, tgt_emb.permute(1, 0, 2),
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt_seq.shape[1]).to(input_vector.device)
            )

            next_token_logits = self.output_layer(output.permute(1, 0, 2))[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # Greedy decoding
            generated_seq.append(next_token)

            if next_token.item() == eos_token:
                break

        return torch.cat(generated_seq, dim=1)


class ConvEmbeddingToSec(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_embedding = ConvEmbedding(config["convolutional_embedding"])
        self.seq_generator = TransformerSeqGen(config["sequence_generator"])

    def forward(self, image, seq):
        b, _, _, _ = image.shape

        emb = (self.conv_embedding(image))
        emb = emb.reshape(b, -1)
        seq = self.seq_generator(emb, seq)

        return seq
