import torch
from torch import nn

from models.networks.FFB_encoder import FFB_encoder
from models.networks.Sine import sine_init


class NFFB(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.xyz_encoder = FFB_encoder(n_input_dims=3, encoding_config=config["encoding"],
                                       network_config=config["SIREN"], has_out=False)
        enc_out_dim = self.xyz_encoder.out_dim

        self.out_lin = nn.Linear(enc_out_dim, 1)

        self.init_output(config["SIREN"]["dims"][-1])


    def init_output(self, layer_size):
        sine_init(self.out_lin, self.xyz_encoder.sin_w0, layer_size)


    def forward(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
        Outputs:
            out: (N), the final sdf value
        """
        out = self.xyz_encoder(x)

        out_feat = torch.cat(out, dim=1)
        out_feat = self.out_lin(out_feat)
        out = out_feat / self.xyz_encoder.grid_level

        return out


    @torch.no_grad()
    # optimizer utils
    def get_params(self, LR_schedulers):
        params = [
            {'params': self.parameters(), 'lr': LR_schedulers[0]["initial"]}
        ]

        return params