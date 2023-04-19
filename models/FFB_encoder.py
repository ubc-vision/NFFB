import torch
import torch.nn as nn

import tinycudann as tcnn
import math

from .Sine import Sine, sine_init, first_layer_sine_init


class FFB_encoder(nn.Module):
    def __init__(self, n_input_dims, network_config, encoding_config, bound):
        super().__init__()

        self.bound = bound

        ### The encoder part
        sin_dims = network_config["SIREN"]["dims"]
        sin_dims = [n_input_dims] + sin_dims  # Plus the input coordinate and latent code
        self.num_sin_layers = len(sin_dims)

        feat_dim = encoding_config["Feat_dim"]
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
        self.ffn_sigma_list = []
        for i in range(grid_level):
            ffn_A = torch.randn((feat_dim, sin_dims[2 + i]), requires_grad=True) # * base_sigma * exp_sigma ** i

            ffn_list.append(ffn_A)

            self.ffn_sigma_list.append(base_sigma * exp_sigma ** i)

        self.register_buffer("ffn_A", torch.stack(ffn_list, dim=0))       ### [grid_level, feat_dim, out_dim]


        ### The low-frequency MLP part
        for layer in range(0, self.num_sin_layers - 1):
            out_dim = sin_dims[layer + 1]
            setattr(self, "sin_lin" + str(layer), nn.Linear(sin_dims[layer], out_dim))

        self.sin_w0 = network_config["SIREN"]["w0"]
        self.sin_activation = Sine(w0=self.sin_w0)
        self.init_siren()


        ### The high-frequency MLP part
        size_factor = network_config["SIREN"]["size_factor"]

        for layer in range(0, grid_level):
            setattr(self, "sin_lin_high" + str(layer), nn.Linear(sin_dims[layer + 1], sin_dims[layer + 2] * size_factor))

        self.sin_w0_high = network_config["SIREN"]["w1"]
        self.sin_activation_high = Sine(w0=self.sin_w0_high)
        self.init_siren_high()

        self.out_dim = sin_dims[-1] * size_factor


    ### Initialize the parameters of SIREN branch
    def init_siren(self):
        for layer in range(0, self.num_sin_layers - 1):
            lin = getattr(self, "sin_lin" + str(layer))

            if layer == 0:
                first_layer_sine_init(lin)
            else:
                sine_init(lin, w0=self.sin_w0)

    def init_siren_high(self):
        for layer in range(0, self.grid_level):
            lin = getattr(self, "sin_lin_high" + str(layer))

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

        ffn_A_list = []
        for i in range(self.grid_level):
            ffn_A_list.append(self.ffn_A[i] * self.ffn_sigma_list[i])
        ffn_A = torch.stack(ffn_A_list, dim=0)

        grid_x = torch.bmm(grid_x, 2 * math.pi * ffn_A)
        grid_x = torch.sin(grid_x)

        x_out = torch.zeros(x.shape[0], self.out_dim, device=in_pos.device)

        ### Grid encoding
        for layer in range(0, self.num_sin_layers - 1):
            sin_lin = getattr(self, "sin_lin" + str(layer))
            x = sin_lin(x)
            x = self.sin_activation(x)

            if layer > 0:
                x = grid_x[layer-1] + x

                sin_lin_high = getattr(self, "sin_lin_high" + str(layer-1))
                x_high = sin_lin_high(x)
                x_high = self.sin_activation_high(x_high)

                x_out = x_out + x_high

        x = x_out

        return x