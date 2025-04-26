import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Tuple, List

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

        self.r_list: List[int] = init_dict["sampling_ratio_list"]
        self.c_list: List[int] = init_dict["channels_list"]
        self.c_o: int = init_dict["num_out_channels"]
        self.f: int = init_dict["proj_filter_size"]


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
        self.proj = nn.Conv2d(self.c_o, self.c_o, kernel_size=self.f, stride=self.f // 4, padding=(self.f - 1) // 2)

        self._freeze_layers()

    def _freeze_layers(self):
        for param in self.proj.parameters():
            param.requires_grad = False

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
        out = self.proj(out)

        return out


class TransformerSeqGen(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config["hidden_dim"]
        vocab_size = config["vocab_size"]
        max_len = config["max_len"]
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        dim_feedforward = config["dim_feedforward"]
        dropout = config["dropout"]

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_embedding_seq, target_ids):
        """
        embedding seq: (b, c, input_dim)
        target_ids: (b, l) - token indices
        """
        b, l = target_ids.size()

        token_emb = self.token_embedding(target_ids)  # (b, l, d_model)
        pos_ids = torch.arange(l, device=target_ids.device).unsqueeze(0).expand(b, l)
        pos_emb = self.pos_embedding(pos_ids)
        tgt = token_emb + pos_emb  # (b, l, d_model)

        tgt_mask = torch.triu(torch.ones(l, l, device=target_ids.device), diagonal=1).bool()

        output = self.decoder(tgt=tgt, memory=input_embedding_seq, tgt_mask=tgt_mask)

        logits = self.output_proj(output)
        return logits

    @torch.no_grad()
    def generate_greedy(self, input_vector, bos_token_idx, eos_token_idx, max_len=20):

        device = input_vector.device
        b = input_vector.size(0)

        input_ids = torch.full((b, 1), bos_token_idx, dtype=torch.long, device=device)

        for _ in range(max_len):
            l = input_ids.size(1)

            tgt = self.token_embedding(input_ids)
            pos_ids = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
            tgt = tgt + self.pos_embedding(pos_ids)

            memory = self.vector_proj(input_vector).unsqueeze(1)  # (b, 1, d_model)
            tgt_mask = torch.triu(torch.ones(l, l, device=device), diagonal=1).bool()

            output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            logits = self.output_proj(output)  # (b, l, vocab_size)

            next_token_logits = logits[:, -1, :]  # (b, vocab_size)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (b, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if (next_token == eos_token_idx).all().item():
                break

        return input_ids

    @torch.no_grad()
    def generate_beam_search(self, input_vector, bos_token_idx, eos_token_idx, max_len=20, beam_width=3):

        device = input_vector.device
        b = input_vector.size(0)
        memory = self.vector_proj(input_vector).unsqueeze(1)  # (b, 1, d_model)

        # Initialize beams: List of lists of (seq, score) tuples per batch item
        beams = [[(torch.full((1, 1), bos_token_idx, dtype=torch.long, device=device), 0.0)] for _ in range(b)]

        finished = [[] for _ in range(b)]  # For storing finished beams (hit EOS)

        for _ in range(max_len):
            new_beams = []

            for i in range(b):
                candidates = []
                for seq, score in beams[i]:
                    if seq[0, -1].item() == eos_token_idx:
                        finished[i].append((seq, score))
                        candidates.append((seq, score))
                        continue

                    l = seq.size(1)
                    tgt = self.token_embedding(seq)
                    pos_ids = torch.arange(l, device=device).unsqueeze(0)
                    tgt = tgt + self.pos_embedding(pos_ids)

                    tgt_mask = torch.triu(torch.ones(l, l, device=device), diagonal=1).bool()
                    mem = memory[i: i + 1]  # (1, 1, d_model)

                    out = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)
                    logits = self.output_proj(out)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

                    topk_log_probs, topk_tokens = torch.topk(log_probs, beam_width)

                    for j in range(beam_width):
                        next_token = topk_tokens[0, j].unsqueeze(0).unsqueeze(0)  # (1, 1)
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_score = score + topk_log_probs[0, j].item()
                        candidates.append((new_seq, new_score))

                # Select top-k candidates for next round
                sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                new_beams.append(sorted_candidates[:beam_width])

            beams = new_beams

            if all(len(f) >= beam_width for f in finished):
                break

        # Finalize output
        output_sequences = []
        for i in range(b):
            final_candidates = finished[i] if finished[i] else beams[i]
            best_seq = sorted(final_candidates, key=lambda x: x[1], reverse=True)[0][0]
            output_sequences.append(best_seq)

        # Pad sequences to same length
        max_out_len = max(seq.size(1) for seq in output_sequences)
        output_tensor = torch.full((b, max_out_len), eos_token_idx, dtype=torch.long, device=device)

        for i, seq in enumerate(output_sequences):
            output_tensor[i, :seq.size(1)] = seq[0]

        return output_tensor


class ConvEmbeddingToSec(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_embedding = ConvEmbedding(config["convolutional_embedding"])
        self.seq_generator = TransformerSeqGen(config["sequence_generator"])

    def forward(self, image, seq):
        emb = (self.conv_embedding(image))
        b, l_emb, _, _ = emb.shape
        emb = emb.reshape(b, l_emb, -1)

        seq = self.seq_generator(emb, seq)

        return seq

    @torch.no_grad()
    def predict(self, image, bos_token_idx, eos_token_idx, max_len=50, beam_search_width: None | int = None):
        b, _, _, _ = image.shape
        emb = (self.conv_embedding(image))
        emb = emb.reshape(b, -1)

        if beam_search_width is None:
            seq = self.seq_generator.generate(emb,
                                              bos_token_idx=bos_token_idx,
                                              eos_token_idx=eos_token_idx,
                                              max_len=max_len)

        else:
            seq = self.seq_generator.generate_beam_search(emb,
                                                          bos_token_idx=bos_token_idx,
                                                          eos_token_idx=eos_token_idx,
                                                          max_len=max_len,
                                                          beam_width=beam_search_width)

        return seq
