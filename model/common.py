from torchvision import transforms
import torch
from torch import nn

use_cuda = torch.cuda.is_available()



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                padding=0, padding_mod='zeros', 
                use_bn = False, use_activation=None):

        super(ConvBlock, self).__init__()
        self.padding = padding
        self.use_bn = use_bn
        self.use_activation = use_activation

        if padding != 0:
            if padding_mod == 'reflect':
                self.pad = nn.ReflectionPad2d(padding)
            elif padding_mod == 'zeros':
                self.pad = nn.ZeroPad2d(padding)
            else:
                raise NotImplementedError(self)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)

        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if self.use_activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif self.use_activation == 'ReLU':
            self.activation = nn.ReLU()
        elif self.use_activation == 'Tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(self)

    def forward(self, x):
        if self.padding != 0:
            x = self.pad(x)

        out = self.conv(x)
        
        if self.use_bn:
            out = self.bn(out)
        if self.use_activation != None:
            out = self.activation(out)

        return out


def get_gradient_by_sobel(input):
    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)

    # Sobel op
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3)
    filter2.weight.data = torch.tensor([
        [-1., -2., -1.],
        [0., 0., 0.],
        [1., 2., 1.]
    ]).reshape(1, 1, 3, 3)
    if use_cuda:
        filter1 = filter1.cuda()
        filter2 = filter2.cuda()

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)

    return image_gradient


def clamp(value, min=0., max=1.0):
    return torch.clamp(value, min=min, max=max)


def RGB2YCbCr(rgb_image):
    """
    :param rgb_image: RGB image data, Tensor: (3, h, w)
    :return: Y, Cb, Cr channel, Tensor: (1, h, w)
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B - Y) * 0.564 + 0.5
    Cr = (R - Y) * 0.713 + 0.5

    Y = clamp(Y)
    Cb = clamp(Cb)
    Cr = clamp(Cr)
    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr):
    """
    :param Y channel, Tensor: (1, h, w)
    :param Cb channel, Tensor: (1, h, w)
    :param Cr channel, Tensor: (1, h, w)
    :return: RGB image, Tensor: (3, h, w)
    """
    
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out


def SAM(inf_batch, vis_batch, threshold):
    """ saliency-aware module """

    # target saliency awareness
    low_freq_image = transforms.GaussianBlur(3)(inf_batch)
    mean_value = torch.mean(inf_batch, (-2, -1), keepdim=True)
    target_saliency_score = torch.abs(low_freq_image - mean_value)
    target_saliency_score = target_saliency_score / torch.amax(target_saliency_score)
    target_saliency_map = torch.where(target_saliency_score > threshold, 
                                torch.ones_like(target_saliency_score), torch.zeros_like(target_saliency_score))

    # gradient saliency awareness
    gradient_inf = get_gradient_by_sobel(inf_batch)
    gradient_vis = get_gradient_by_sobel(vis_batch)
    grad_saliency_map_inf = torch.where(gradient_inf > gradient_vis, torch.ones_like(gradient_inf), torch.zeros_like(gradient_inf))
    grad_saliency_map_vis = torch.ones_like(grad_saliency_map_inf) - grad_saliency_map_inf

    return target_saliency_map, grad_saliency_map_inf, grad_saliency_map_vis, \
            gradient_inf, gradient_vis



if __name__ == '__main__':
    ...
