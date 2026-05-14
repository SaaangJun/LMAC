import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator_net(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(Discriminator_net, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        
        self.hidden_dim = getattr(args, "hidden_dim", 64)
        self.latent_dim = getattr(args, "latent_dim", 20)
        self.dropout_rate = getattr(args, "dropout", 0.1)

        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.latent_fc = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, inputs, hidden_state):
        
        hidden_state = hidden_state.reshape(-1, self.hidden_dim)
        
        x = self.encoder_fc(inputs)
        h = self.gru(x, hidden_state)
        z = self.latent_fc(h) 

        out = self.decoder(z)
        
        return out, z, h

    def init_hidden(self):
        return self.encoder_fc[0].weight.new_zeros(1, self.hidden_dim)
