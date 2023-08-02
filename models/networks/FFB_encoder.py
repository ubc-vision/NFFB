import torch
import torch.nn as nn

import tinycudann as tcnn
import math

from models.networks.Sine import Sine, sine_init, first_layer_sine_init


class FFB_encoder(nn.Module):
    def __init__(self, encoding_config, network_config, n_input_dims, bound=1.0, has_out=True):
        super().__init__()

        self.bound = bound

        ### The encoder part
        sin_dims = network_config["dims"]
        sin_dims = [n_input_dims] + sin_dims
        self.num_sin_layers = len(sin_dims)

        feat_dim = encoding_config["feat_dim"]
        base_resolution = encoding_config["base_resolution"]
        per_level_scale = encoding_config["per_level_scale"]

        assert self.num_sin_layers > 3, "The layer number (SIREN branch) should be greater than 3."
        grid_level = int(self.num_sin_layers - 2)
        self.grid_encoder = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": grid_level,
                "n_features_per_level": feat_dim,
                "log2_hashmap_size": 19,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.grid_level = grid_level
        print(f"Grid encoder levels: {grid_level}")

        self.feat_dim = feat_dim

        ### Create the ffn to map low-dim grid feats to map high-dim SIREN feats
        base_sigma = encoding_config["base_sigma"]
        exp_sigma = encoding_config["exp_sigma"]

        ffn_list = []
        for i in range(grid_level):
            ffn = torch.randn((feat_dim, sin_dims[2 + i]), requires_grad=True) * base_sigma * exp_sigma ** i

            ffn_list.append(ffn)

        self.ffn = nn.Parameter(torch.stack(ffn_list, dim=0))


        ### The low-frequency MLP part
        for layer in range(0, self.num_sin_layers - 1):
            setattr(self, "sin_lin" + str(layer), nn.Linear(sin_dims[layer], sin_dims[layer + 1]))

        self.sin_w0 = network_config["w0"]
        self.sin_activation = Sine(w0=self.sin_w0)
        self.init_siren()

        ### The output layers
        self.has_out = has_out
        if has_out:
            size_factor = network_config["size_factor"]
            self.out_dim = sin_dims[-1] * size_factor

            for layer in range(0, grid_level):
                setattr(self, "out_lin" + str(layer), nn.Linear(sin_dims[layer + 1], self.out_dim))

            self.sin_w0_high = network_config["w1"]
            self.init_siren_out()
            self.out_activation = Sine(w0=self.sin_w0_high)
        else:
            self.out_dim = sin_dims[-1] * grid_level


    ### Initialize the parameters of SIREN branch
    def init_siren(self):
        for layer in range(0, self.num_sin_layers - 1):
            lin = getattr(self, "sin_lin" + str(layer))

            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin, w0=self.sin_w0)


    def init_siren_out(self):
        for layer in range(0, self.grid_level):
            lin = getattr(self, "out_lin" + str(layer))

            sine_init(lin, w0=self.sin_w0_high)


    def forward(self, in_pos):
        """
        in_pos: [N, 3], in [-bound, bound]

        in_pos (for grid features) should always be located in [0.0, 1.0]
        x (for SIREN branch) should always be located in [-1.0, 1.0]
        """

        x = in_pos / self.bound								# to [-1, 1]
        in_pos = (in_pos + self.bound) / (2 * self.bound) 	# to [0, 1]

        grid_x = self.grid_encoder(in_pos)
        grid_x = grid_x.view(-1, self.grid_level, self.feat_dim)
        grid_x = grid_x.permute(1, 0, 2)

        embedding_list = []
        for i in range(self.grid_level):
            grid_output = torch.matmul(grid_x[i], self.ffn[i])
            grid_output = torch.sin(2 * math.pi * grid_output)
            embedding_list.append(grid_output)

        if self.has_out:
            x_out = torch.zeros(x.shape[0], self.out_dim, device=in_pos.device)
        else:
            feat_list = []

        ### Grid encoding
        for layer in range(0, self.num_sin_layers - 1):
            sin_lin = getattr(self, "sin_lin" + str(layer))
            x = sin_lin(x)
            x = self.sin_activation(x)

            if layer > 0:
                x = embedding_list[layer-1] + x

                if self.has_out:
                    out_lin = getattr(self, "out_lin" + str(layer-1))
                    x_high = out_lin(x)
                    x_high = self.out_activation(x_high)

                    x_out = x_out + x_high
                else:
                    feat_list.append(x)

        if self.has_out:
            x = x_out
        else:
            x = feat_list

        return x