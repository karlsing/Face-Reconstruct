from torch import nn

class DGWGAN(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super(DGWGAN, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                #nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),
            nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            nn.Conv2d(dim*2, 1, 32),
            #conv_ln_lrelu(dim * 4, dim * 8),
            #nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid())

    def forward(self, x):
        y = self.ls(x.float())
        y = y.view(-1)
        return y