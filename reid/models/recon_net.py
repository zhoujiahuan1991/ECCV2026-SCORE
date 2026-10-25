import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter

__all__ = ['RBTBlock']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class RBTBlock(nn.Module):
    """
    The RBT block introduced in <Unified Representation Learning for Cross Model Compatibility>
    """
    def __init__(self, in_planes=2048, out_planes=2048, num_paths=4):
        super(RBTBlock, self).__init__()
        if not isinstance(num_paths, int) or not 0 <= num_paths <= 4:
            raise ValueError('num_paths: {}'.format(num_paths))
        self.num_paths = num_paths
        if self.num_paths == 0:
            print('No need to construct trans.path since num_paths is 0.')
        for i in range(self.num_paths):
            print('Construct trans.path{}: {} -> {}'.format(i + 1, in_planes, out_planes))
            setattr(self, 'path{}'.format(i + 1), self._make_onepath(in_planes, out_planes))

    def _make_onepath(self, in_planes, out_planes):
        return nn.Sequential(nn.Linear(in_planes, 32, bias=False),
                             nn.BatchNorm1d(32, eps=2e-05, momentum=0.9),
                             nn.PReLU(32),
                             nn.Linear(32, 32, bias=False),
                             nn.BatchNorm1d(32, eps=2e-05, momentum=0.9),
                             nn.PReLU(32),
                             nn.Linear(32, out_planes, bias=False),
                             nn.BatchNorm1d(out_planes, eps=2e-05, momentum=0.9),
                             nn.PReLU(out_planes)
                             )

    def forward(self, feat):
        output = feat

        # transnet
        for i in range(self.num_paths):
            output = output + getattr(self, 'path{}'.format(i + 1))(feat)

        return output
    
class ReconNet(nn.Module):
    def __init__(self, in_planes=2048, out_planes=2048, num_paths=4):
        super(ReconNet, self).__init__()
        
        self.trans_forward = nn.Sequential(
			RBTBlock(in_planes, out_planes, num_paths),
			RBTBlock(in_planes, out_planes, num_paths),
			RBTBlock(in_planes, out_planes, num_paths),
			RBTBlock(in_planes, out_planes, num_paths))
        
        self.trans_forward.apply(weights_init_kaiming)


    def forward(self, x):
        x = self.trans_forward(x)
        
        return x