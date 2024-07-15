import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Pident2(nn.Module):
    def __init__(
        self,
        params_in=None,
        in_channels=(3, 4),
        device=None,
        attn_map=False,
        batch_first=False,
    ):
        super().__init__()
        self.init_params(params_in)
        d_model = self.params["d_model"]
        d_model_2d = self.params["d_model_2d"]
        d_model_conn = self.params["d_model_conn"]
        self.nhead = self.params["nhead"]
        dropout_p = self.params["dropout_p"]
        inter_scale = self.params["inter_scale"]
        inter_scale_2d = self.params["inter_scale_2d"]
        encoder_layers = self.params["encoder_layers"]
        self.attn_gate = self.params["attn_gate"]
        self.attn_bias = self.params["attn_bias"]
        self.attn_map = attn_map
        self.batch_first = batch_first

        decoder_layers = [d_model_2d, d_model_2d // 2, d_model_2d // 4]
        if d_model % self.nhead != 0:
            raise ValueError("nhead needs to be a factor of d_model")
        self.head_dim = d_model // self.nhead
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)

        self.in_layer = nn.Linear(in_channels[0], d_model)
        self.in_layer_2d = nn.Linear(in_channels[1], d_model_2d)
        self.dropout = nn.Dropout(dropout_p)
        self.encoder = nn.ModuleList()
        self.proj_weights = Parameter(
            torch.empty(
                (encoder_layers, d_model * 3, d_model),
                device=device,
                dtype=torch.float32,
            )
        )

        for i in range(encoder_layers):
            encoder_block = nn.ModuleDict(
                {
                    "tr_norm": nn.LayerNorm(d_model),
                    "attn_out": nn.Linear(d_model, d_model),
                    "dropout": nn.Dropout(dropout_p),
                }
            )
            if self.attn_bias:
                encoder_block["bias"] = nn.Linear(d_model_2d, self.nhead, bias=False)
                encoder_block["bias_norm"] = nn.LayerNorm(d_model_2d)
            if self.attn_gate:
                encoder_block["gate"] = nn.Linear(d_model, d_model)

            encoder_block["linear_1d"] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * inter_scale),
                nn.ReLU(inplace=True),
                nn.Linear(d_model * inter_scale, d_model),
            )

            encoder_block["conn_norm"] = nn.LayerNorm(d_model)
            encoder_block["conn_1d-lin"] = nn.Linear(d_model, d_model_conn)
            encoder_block["conn_2d-lin"] = nn.Linear(d_model_conn * 2, d_model_2d)

            encoder_block["linear_2d"] = nn.Sequential(
                nn.LayerNorm(d_model_2d),
                nn.Linear(d_model_2d, d_model_2d * inter_scale_2d),
                nn.ReLU(inplace=True),
                nn.Linear(d_model_2d * inter_scale_2d, d_model_2d),
            )
            self.encoder.append(encoder_block)

        self.decoder_norm = nn.LayerNorm(d_model_2d)
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_layers) - 1):
            self.decoder.append(nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
            self.decoder.append(nn.ReLU(inplace=True))
        self.out_layer = nn.Linear(decoder_layers[-1], 1)
        self.init_weights()

    def init_params(self, params_in):
        params = {
            "d_model": 32,
            "d_model_2d": 32,
            "d_model_conn": 16,
            "nhead": 8,
            "dropout_p": 0.1,
            "inter_scale": 2,
            "inter_scale_2d": 4,
            "encoder_layers": 12,
            "attn_gate": False,
            "attn_bias": True,
        }
        param_set = set(params.keys())
        if params_in != None:
            in_set = set(params_in.keys())
            if len(param_set.union(in_set)) != len(param_set):
                unmatched = (param_set ^ in_set) & in_set
                raise RuntimeError("Unmatched model parameters: " + str(unmatched))
            params.update(params_in)
        self.params = params

    def get_params(self):
        return self.params

    def init_weights(self):
        torch.nn.init.trunc_normal_(self.in_layer.weight)
        torch.nn.init.zeros_(self.in_layer.bias)
        torch.nn.init.trunc_normal_(self.in_layer_2d.weight)
        torch.nn.init.zeros_(self.in_layer_2d.bias)
        torch.nn.init.xavier_uniform_(self.proj_weights)

        for encoder_block in self.encoder:
            if self.attn_gate:
                torch.nn.init.zeros_(encoder_block["gate"].weight)
                torch.nn.init.ones_(encoder_block["gate"].bias)

            torch.nn.init.zeros_(encoder_block["attn_out"].weight)
            torch.nn.init.zeros_(encoder_block["attn_out"].bias)
            torch.nn.init.trunc_normal_(encoder_block["bias"].weight)

            torch.nn.init.trunc_normal_(encoder_block["linear_1d"][1].weight)
            torch.nn.init.zeros_(encoder_block["linear_1d"][1].bias)
            torch.nn.init.zeros_(encoder_block["linear_1d"][3].weight)
            torch.nn.init.zeros_(encoder_block["linear_1d"][3].bias)

            torch.nn.init.trunc_normal_(encoder_block["conn_1d-lin"].weight)
            torch.nn.init.zeros_(encoder_block["conn_1d-lin"].bias)
            torch.nn.init.trunc_normal_(encoder_block["conn_2d-lin"].weight)
            torch.nn.init.zeros_(encoder_block["conn_2d-lin"].bias)

            torch.nn.init.trunc_normal_(encoder_block["linear_2d"][1].weight)
            torch.nn.init.zeros_(encoder_block["linear_2d"][1].bias)
            torch.nn.init.zeros_(encoder_block["linear_2d"][3].weight)
            torch.nn.init.zeros_(encoder_block["linear_2d"][3].bias)

        for layer in self.decoder:
            if layer.__class__ == nn.Linear:
                torch.nn.init.trunc_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.out_layer.weight)
        torch.nn.init.zeros_(self.out_layer.bias)

    def self_attn(self, x_1d, x_2d, encoder_block, proj_weight, pad_mask=None):
        src_len = x_1d.size(0)
        bsz = x_1d.size(1)
        q, k, v = F.linear(x_1d, proj_weight).chunk(3, dim=-1)
        proj_shape = (src_len, bsz * self.nhead, self.head_dim)
        q = q.reshape(proj_shape).transpose(0, 1)
        k = k.reshape(proj_shape).transpose(0, 1)
        v = v.reshape(proj_shape).transpose(0, 1)
        if self.attn_gate:
            gate = torch.sigmoid(encoder_block["gate"](x_1d))
            gate = gate.reshape(proj_shape).transpose(0, 1)

        if pad_mask != None:
            attn_mask = (
                pad_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.nhead, -1, -1)
                .reshape(bsz * self.nhead, 1, src_len)
            )
            x_attn = torch.baddbmm(
                attn_mask, q, k.transpose(1, 2)
            )  # (bsz * head, length_r, length_c)
        else:
            x_attn = torch.bmm(q, k.transpose(1, 2))
        x_attn *= self.scale_factor

        if self.attn_bias:
            bias = torch.clone(x_2d)
            bias = encoder_block["bias_norm"](bias)
            bias = encoder_block["bias"](bias)  # (bsz, length_c, length_w, head)
            x_attn += bias.permute(0, 3, 1, 2).flatten(
                end_dim=1
            )  # (bsz * head, length_r, length_c)

        x_attn = F.softmax(x_attn, dim=-1)
        x_attn = self.dropout(x_attn)

        if self.attn_map:
            layer_attn = list()
            t_index = 0
            for i in range(bsz):
                batch_attn = list()
                for j in range(self.nhead):
                    head_attn = dict()
                    head_attn["attn"] = x_attn[t_index].detach().cpu().numpy()
                    head_attn["bias"] = bias[:, :, t_index].detach().cpu().numpy()
                    batch_attn.append(head_attn)
                    t_index += 1
                layer_attn.append(batch_attn)
        else:
            layer_attn = None

        x_attn = torch.bmm(x_attn, v)
        if self.attn_gate:
            x_attn *= gate
        x_attn = x_attn.transpose(0, 1).contiguous().view(src_len * bsz, x_1d.size(2))
        x_attn = encoder_block["attn_out"](x_attn)
        x_attn = encoder_block["dropout"](x_attn)
        x_attn = x_attn.view(src_len, bsz, x_1d.size(2))
        return x_attn, layer_attn

    def forward(self, x_1d, x_2d, pad_mask=None, self_attn=True):
        if self.batch_first:
            x_1d = x_1d.transpose(0, 1)

        bsz = x_1d.size(1)
        if self_attn:
            if self.attn_map:
                attn_map = [list() for i in range(bsz)]
            else:
                attn_map = None

            x_1d = self.in_layer(x_1d)
        x_2d = self.in_layer_2d(x_2d)

        for i, encoder_block in enumerate(self.encoder):
            if self_attn:
                identity_1d = torch.clone(x_1d)
                x_1d = encoder_block["tr_norm"](x_1d)
                x_1d, layer_attn = self.self_attn(
                    x_1d, x_2d, encoder_block, self.proj_weights[i], pad_mask
                )
                x_1d += identity_1d

                if self.attn_map:
                    for j in range(bsz):
                        attn_map[j].append(layer_attn[j])

                identity_1d = torch.clone(x_1d)
                x_1d = encoder_block["linear_1d"](x_1d)
                x_1d += identity_1d

                x_conn = encoder_block["conn_norm"](x_1d)
                x_conn = encoder_block["conn_1d-lin"](x_conn)
                x_conn = self.outer(x_conn)
                x_conn = encoder_block["conn_2d-lin"](x_conn)
                x_2d += x_conn

            identity_2d = torch.clone(x_2d)
            x_2d = encoder_block["linear_2d"](x_2d)
            x_2d += identity_2d

        x_2d = self.decoder_norm(x_2d)
        for layer in self.decoder:
            x_2d = layer(x_2d)
        x_2d = self.out_layer(x_2d).squeeze(dim=-1)

        if not self.attn_map:
            return x_2d
        else:
            return x_2d, attn_map

    def outer(self, x):
        in_size = x.size()
        target_size = (in_size[1], in_size[0], in_size[0], in_size[2])
        x_o = x.transpose(0, 1).unsqueeze(dim=2).expand(target_size)
        x_o = torch.cat((x_o, x_o.transpose(1, 2)), dim=-1)
        return x_o


class TestModel(nn.Module):
    def __init__(self, params_in=None, in_features=(3, 4), device=None):
        super().__init__()
        self.init_params(params_in)
        d_model = self.params["d_model"]
        dropout_p = self.params["dropout_p"]
        layers = self.params["encoder_layers"]
        self.max_len = 10.0

        self.in_layer = nn.Linear(in_features[1], d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

        self.encoder = nn.ModuleList()
        for i in range(layers):
            self.encoder.append(nn.Linear(d_model, d_model))
            self.encoder.append(nn.LayerNorm(d_model))
            self.encoder.append(nn.ReLU(True))
            self.encoder.append(nn.Dropout(dropout_p))

        self.out_layer = nn.Linear(d_model, 1)

    def init_params(self, params_in):
        params = {
            "d_model": 64,
            "dropout_p": 0.1,
            "encoder_layers": 4,
        }
        param_set = set(params.keys())
        if params_in != None:
            in_set = set(params_in.keys())
            if len(param_set.union(in_set)) != len(param_set):
                unmatched = (param_set ^ in_set) & in_set
                raise RuntimeError("Unmatched model parameters: " + str(unmatched))
            params.update(params_in)
        self.params = params

    def get_params(self):
        return self.params

    def forward(self, x_1d, x_2d, pad_mask=None, self_attn=True):
        x_2d = self.in_layer(x_2d)
        x_2d = self.in_norm(x_2d)
        x_2d = self.dropout(x_2d)

        for layer in self.encoder:
            x_2d = layer(x_2d)
        x_2d = self.out_layer(x_2d)
        return x_2d.squeeze(dim=-1)


pident_models = {"pident2": Pident2, "test": TestModel}
