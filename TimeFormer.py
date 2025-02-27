import torch
from torch import nn
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


class TimeFormer(nn.Module):

    def __init__(self, input_dims=4, multires=4, nhead=6, hidden_dims=48, num_layer=3):
        super(TimeFormer, self).__init__()
        self.embed_fn, self.embed_ch = get_embedder(multires=multires, input_dims=input_dims)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_ch, nhead=nhead,
                                                          dim_feedforward=hidden_dims,
                                                          activation=nn.functional.tanh)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        self.output_layer = nn.Linear(self.embed_ch, 3)
        
        self.spatial_lr_scale = 5 

        
    def train_setting(self, training_args):
        l = [
            {'params': list(self.transformer_encoder.parameters()), 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "deform_encoder"},
            {'params': list(self.output_layer.parameters()),'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "timeformer_decoder"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            # if ''param_group["name"] == "":
            lr = self.deform_scheduler_args(iteration)
            param_group['lr'] = lr
            return lr
        
        

    def forward(self, src):
        src_embed = self.embed_fn(src)
        h = self.transformer_encoder(src_embed)
        h = self.output_layer(h)
        return h


if __name__ == '__main__':
    src = torch.rand((10, 110, 4))

    timeFormer = TimeFormer(input_dims=4, multires=4, nhead=6)

    out = timeFormer(src)

    print(out.shape)
