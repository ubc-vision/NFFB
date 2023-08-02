import torch
from torch import nn

from models.networks.FFB_encoder import FFB_encoder


class NFFB(nn.Module):
    def __init__(self, config, out_dims=3):
        super().__init__()

        self.xyz_encoder = FFB_encoder(n_input_dims=2, encoding_config=config["encoding"],
                                       network_config=config["SIREN"], has_out=False)

        ### Initializing backbone part, to merge multi-scale grid features
        backbone_dims = config["Backbone"]["dims"]
        grid_feat_len = self.xyz_encoder.out_dim
        backbone_dims = [grid_feat_len] + backbone_dims + [out_dims]
        self.num_backbone_layers = len(backbone_dims)

        for layer in range(0, self.num_backbone_layers - 1):
            out_dim = backbone_dims[layer + 1]
            setattr(self, "backbone_lin" + str(layer), nn.Linear(backbone_dims[layer], out_dim))

        self.relu_activation = nn.ReLU(inplace=True)


    @torch.no_grad()
    # optimizer utils
    def get_params(self, LR_schedulers):
        params = [
            {'params': self.parameters(), 'lr': LR_schedulers[0]["initial"]}
        ]

        return params


    def forward(self, in_pos):
        """
        Inputs:
            x: (N, 2) xy in [-scale, scale]
        Outputs:
            out: (N, 1 or 3), the RGB values
        """
        x = (in_pos - 0.5) * 2.0

        grid_x = self.xyz_encoder(x)
        out_feat = torch.cat(grid_x, dim=1)


        ### Backbone transformation
        for layer in range(0, self.num_backbone_layers - 1):
            backbone_lin = getattr(self, "backbone_lin" + str(layer))
            out_feat = backbone_lin(out_feat)

            if layer < self.num_backbone_layers - 2:
                out_feat = self.relu_activation(out_feat)

        out_feat = out_feat.clamp(-1.0, 1.0)

        return out_feat