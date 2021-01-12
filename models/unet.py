from module.Unet_parts import *
from module.Attention_block import weights_init_normal


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, downsize_nb_filters_factor=2):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.down1 = down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.down2 = down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.down3 = down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.down4 = down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.up1 = up(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.up2 = up(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.up3 = up(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.up4 = up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.outc = outconv(64 // downsize_nb_filters_factor, n_classes)
        self.apply(weights_init_normal)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x


class Kumar_Net(nn.Module):
    def __init__(self, n_channels, n_classes, d=64):
        super(Kumar_Net, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv3d(n_channels, d, 4, 2, 1)
        self.conv2 = nn.Conv3d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm3d(d * 2)
        self.conv3 = nn.Conv3d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d(d * 4)
        self.conv4 = nn.Conv3d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm3d(d * 8)
        self.conv5 = nn.Conv3d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm3d(d * 8)

        # Unet decoder
        self.deconv1 = nn.ConvTranspose3d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm3d(d * 8)
        self.deconv2 = nn.ConvTranspose3d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm3d(d * 8)
        self.deconv3 = nn.ConvTranspose3d(d * (8 + 4), d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm3d(d * 8)
        self.deconv4 = nn.ConvTranspose3d(d * (8 + 2), d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm3d(d * 8)
        self.deconv5 = nn.ConvTranspose3d(d * (8 + 1), n_classes, 4, 2, 1)
        self.apply(weights_init_normal)

    def forward(self, input, masks):
        e1 = self.conv1(input)
        e2 = F.leaky_relu(F.dropout(self.conv2_bn(self.conv2(e1)), 0.5), 0.2)
        e3 = F.leaky_relu(F.dropout(self.conv3_bn(self.conv3(e2)), 0.5), 0.2)
        e4 = F.leaky_relu(F.dropout(self.conv4_bn(self.conv4(e3)), 0.5), 0.2)
        e5 = F.leaky_relu(F.dropout(self.conv5_bn(self.conv5(e4)), 0.5), 0.2)
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e5))), 0.5, training=True)
        d1 = torch.cat([d1, e4], 1)

        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e3], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e2], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e1], 1)
        d5 = self.deconv5(F.relu(d4))
        x = torch.sigmoid(d5)
        out = self.postprocess(input, masks, x)
        return out

    def postprocess(self, masked_imgs, masks, c11):
        inpainted = c11 * masks
        out = inpainted + masked_imgs * (1 - masks)
        return out


class Dakai_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels, n_classes, n1=64):
        super(Dakai_Net, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()
    def postprocess(self, masked_imgs, masks, c11):
        inpainted = c11 * masks
        out = inpainted + masked_imgs * (1 - masks)
        return out

    def forward(self, input, masks):
        e1 = self.Conv1(input)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = torch.sigmoid(self.Conv(d2))
        out = self.postprocess(input, masks, out)
        return out


