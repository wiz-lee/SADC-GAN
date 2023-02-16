import torch
import torch.nn as nn
from model.common import ConvBlock, clamp



class FFM(nn.Module):
    def __init__(self, in_channels, pre_channels, out_channels):
        super(FFM, self).__init__()

        self.attention_inf = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.attention_vis = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid_inf = nn.Sigmoid()
        self.sigmoid_vis = nn.Sigmoid()

        self.soft_max = nn.Softmax()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                stride=1, padding=1, padding_mod='reflect', use_bn=True, use_activation='LeakyReLU')
        self.conv2 = ConvBlock(in_channels=in_channels+pre_channels, out_channels=out_channels, kernel_size=3,
                                stride=1, padding=1, padding_mod='reflect', use_bn=True, use_activation='LeakyReLU')

    def forward(self, inf_batch_feats, vis_batch_feats, pre_batch_feats=None):
        # spatial attention
        avg_inf = torch.mean(inf_batch_feats, dim=1, keepdim=True)
        avg_vis = torch.mean(vis_batch_feats, dim=1, keepdim=True)
        max_inf, _ = torch.max(inf_batch_feats, dim=1, keepdim=True) 
        max_vis, _ = torch.max(vis_batch_feats, dim=1, keepdim=True)
        x_inf = torch.cat([avg_inf, max_inf], dim=1)
        x_vis = torch.cat([avg_vis, max_vis], dim=1)
        score_inf = self.attention_inf(x_inf) # (b, 2, h, w) to (b, 1, h, w)
        score_vis = self.attention_vis(x_vis)
        w_inf = self.sigmoid_inf(score_inf)
        w_vis = self.sigmoid_vis(score_vis)

        # layered and progressive feature fusion
        w = torch.cat([w_inf, w_vis], dim=1)
        w = torch.softmax(w, dim=1)
        w_inf = w[:, 0:1, :, :]
        w_vis = w[:, 1:2, :, :]
        fused_feats = w_inf * inf_batch_feats + w_vis * vis_batch_feats # (b, in_channels, h, w)

        out = self.conv1(fused_feats) # (b, in_channels, h, w) to (b, in_channels, h, w)

        if pre_batch_feats != None:
            out = torch.cat([pre_batch_feats, out], dim=1) # to (b, in_channels + pre_channels, h, w)
        out = self.conv2(out) # to (b, out_channels, h, w)
        
        return out


class ExtractionAndFusionNetwork(nn.Module):
    def __init__(self):
        super(ExtractionAndFusionNetwork, self).__init__()

        self.conv1_inf = ConvBlock(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0, padding_mod='zeros', 
                    use_bn=True, use_activation='LeakyReLU') 
        self.conv1_vis = ConvBlock(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0, padding_mod='zeros', 
                    use_bn=True, use_activation='LeakyReLU') 
        self.FFM1 = FFM(16, 0, 32)

        self.conv2_inf = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv2_vis = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.FFM2 = FFM(32, 32, 64)

        self.conv3_inf = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv3_vis = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.FFM3 = FFM(64, 64, 128)

        self.conv4_inf = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv4_vis = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.FFM4 = FFM(128, 128, 256)

    def forward(self, inf_batch, vis_batch):
        out1_inf = self.conv1_inf(inf_batch)
        out1_vis = self.conv1_vis(vis_batch)
        fuse1 = self.FFM1(out1_inf, out1_vis, None) # 16 + 0 -> 32

        out2_inf = self.conv2_inf(out1_inf)
        out2_vis = self.conv2_vis(out1_vis)
        fuse2 = self.FFM2(out2_inf, out2_vis, fuse1) # 32 + 32 -> 64

        out3_inf = self.conv3_inf(out2_inf)
        out3_vis = self.conv3_vis(out2_vis)
        fuse3 = self.FFM3(out3_inf, out3_vis, fuse2) # 64 + 64 -> 128

        out4_inf = self.conv4_inf(out3_inf)
        out4_vis = self.conv4_vis(out3_vis)
        fuse4 = self.FFM4(out4_inf, out4_vis, fuse3) # 128 + 128 -> 256

        return fuse4


class ReconstructionNetwork(nn.Module):
    def __init__(self):
        super(ReconstructionNetwork, self).__init__()
        
        self.conv1 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv2 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv3 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv4 = ConvBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv5 = ConvBlock(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, padding_mod='zeros',
                    use_bn=True, use_activation='Tanh')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) # is (b, 1, h, w)
        x = x / 2 + 0.5  # [-1,1] to [0,1]
        return x


class DecompositionNetwork(nn.Module):
    def __init__(self):
        super(DecompositionNetwork, self).__init__()
        self.conv1 = ConvBlock(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0, padding_mod='zeros', 
                    use_bn=True, use_activation='LeakyReLU') 

        self.conv2_inf = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv2_vis = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')

        self.conv3_inf = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv3_vis = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')       

        self.conv4_inf = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv4_vis = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
    
        self.conv5_inf = ConvBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv5_vis = ConvBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')

        self.conv6_inf = ConvBlock(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, padding_mod='zeros', 
                    use_bn=True, use_activation='Tanh')
        self.conv6_vis = ConvBlock(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, padding_mod='zeros', 
                    use_bn=True, use_activation='Tanh')

    def forward(self, fused_img):
        out1_inf = out1_vis = self.conv1(fused_img)

        out2_inf = self.conv2_inf(out1_inf)
        out2_vis = self.conv2_vis(out1_vis)
        out3_inf = self.conv3_inf(out2_inf)
        out3_vis = self.conv3_vis(out2_vis)
        out4_inf = self.conv4_inf(out3_inf)
        out4_vis = self.conv4_vis(out3_vis)
        out5_inf = self.conv5_inf(out4_inf)
        out5_vis = self.conv5_vis(out4_vis)
        out6_inf = self.conv6_inf(out5_inf)
        out6_vis = self.conv6_vis(out5_vis)
        out6_inf = out6_inf / 2 + 0.5
        out6_vis = out6_vis / 2 + 0.5


        return out6_inf, out6_vis # fake_inf, fake_vis



class GeneratorTrainPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor_fuser = ExtractionAndFusionNetwork()
        self.fr = ReconstructionNetwork()
        self.dcnet = DecompositionNetwork()

    def forward(self, inf_batch, vis_batch):
        fused_features = self.extractor_fuser(inf_batch, vis_batch) 
        fused_image = self.fr(fused_features)
        fused_image = clamp(fused_image)

        fake_inf, fake_vis = self.dcnet(fused_image)
        fake_inf = clamp(fake_inf)
        fake_vis = clamp(fake_vis)

        return fused_image, fake_inf, fake_vis



class GeneratorTestPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor_fuser = ExtractionAndFusionNetwork()
        self.fr = ReconstructionNetwork()

    def forward(self, inf_batch, vis_batch):
        fused_features = self.extractor_fuser(inf_batch, vis_batch) 
        fused_image = self.fr(fused_features)
        fused_image = clamp(fused_image)

        return fused_image



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, padding_mod='reflect', 
                    use_bn=False, use_activation='LeakyReLU')
        self.conv2 = ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv3 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv4 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')
        self.conv5 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, padding_mod='reflect', 
                    use_bn=True, use_activation='LeakyReLU')

        self.fc = nn.Linear(1024, 1)
        self.activation = nn.Tanh()


    def forward(self, batch_img):
        out = self.conv1(batch_img)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.shape[0], -1) # (b, c, h, w) => (b, c*h*w)
        
        predicted_value = self.fc(out)
        y = self.activation(predicted_value)
        y = y / 2 + 0.5
        return y


class SADC_GAN(nn.Module):
    def __init__(self):
        super(SADC_GAN, self).__init__()
        self.generator = GeneratorTrainPhase()
        self.discriminator_inf = Discriminator()
        self.discriminator_vis = Discriminator()

    def forward(self, inf_batch, vis_batch):
        fused_image, fake_inf, fake_vis = self.generator(inf_batch, vis_batch)
        predicted_value_inf_fake = self.discriminator_inf(fake_inf)
        predicted_value_inf_real = self.discriminator_inf(inf_batch)

        predicted_value_vis_fake = self.discriminator_vis(fake_vis)
        predicted_value_vis_real = self.discriminator_vis(vis_batch)

        return fused_image, fake_inf, fake_vis, predicted_value_inf_real, predicted_value_inf_fake, \
            predicted_value_vis_real, predicted_value_vis_fake




if __name__ == '__main__':
    ...
