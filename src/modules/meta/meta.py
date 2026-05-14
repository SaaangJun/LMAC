import torch
import torch.nn as nn
import torch.nn.functional as F

class meta(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(meta, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hidden_dim = args.hidden_dim
        self.latent_dim = self.args.latent_dim
        self.output_dim = output_dim
        dr = getattr(args, "dropout", 0.1)

        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dr)
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.q_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_layer = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim * 2) 
        )
        self.recon_encoder = nn.Sequential(
            nn.Linear(output_dim * 2, self.hidden_dim),  
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(self.hidden_dim, self.latent_dim) 
        )

    def forward(self, inputs, hidden_state):
        batch_total = inputs.shape[0]
        hidden_state = hidden_state.reshape(-1, self.hidden_dim)

        x = self.encoder_mlp(inputs)         
        h = self.gru(x, hidden_state)        
        h = h + x                           

        q = self.q_layer(x)
        k = self.k_layer(h)
        v = self.v_layer(h)
        attn_score = (q * k).sum(-1, keepdim=True)
        attn_weight = torch.sigmoid(attn_score)
        latent_z = attn_weight * v           

        out = self.decoder(latent_z)         
        
        pred_state, pred_meta = out.split(self.output_dim, dim=-1)
        pred_meta = torch.sigmoid(pred_meta)  
        
        out_combined = torch.cat([pred_state, pred_meta], dim=-1)
        
        latent_recon = self.recon_encoder(out_combined)  

        return out_combined, latent_z, h, latent_recon

    def init_hidden(self):
        return self.encoder_mlp[0].weight.new_zeros(1, self.hidden_dim)

