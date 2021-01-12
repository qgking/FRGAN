from tumor_data.SYNDataLoader import *
from tensorboardX import SummaryWriter
from torchsummaryX import summary
from driver.helper.base_syn_helper import BaseTrainHelper


class SYNHelper_Dilate_Gate_Richer_Lcon_Lpercep_Lsty(BaseTrainHelper):
    def __init__(self, generator, discriminator,
                 criterions, config):
        super(SYNHelper_Dilate_Gate_Richer_Lcon_Lpercep_Lsty, self).__init__(generator, discriminator,
                                                                             criterions, config)

    def out_put_shape(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)
        # summary(self.generator.cpu(),
        #         torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)),
        #         torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)))
        # summary(self.discriminator.cpu(),
        #         torch.zeros((1, 2, self.config.patch_x, self.config.patch_y, self.config.patch_z)))
        # summary(self.generator.cpu(),
        #         torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)),
        #         torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)))
        # summary(self.discriminator.cpu(),gatedilatericher
        #         torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)))
        # print(self.generator)
        # print(self.discriminator)
        # g_out = self.generator(torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)),
        #                        torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)))
        # fake_B = g_out['outputs']
        # boundary_out = g_out['so_out']

    def test_one_batch(self, batch_gen):
        real_A = batch_gen['real_A']
        real_B = batch_gen['real_B']
        boundary_B = batch_gen['boundary_B']
        tumor_B = batch_gen['tumor_B']

        g_out = self.generator(real_A, tumor_B)
        fake_B = g_out['outputs']
        boundary_out = g_out['so_out']
        fake_B_coarse = g_out['coarse_outputs']
        boundary_out_coarse = g_out['so_out_c']

        # loss style
        losses_style = 0
        if 'Lsty' in self.config.helper:
            loss_style = self.criterions['StyleLoss'](fake_B, real_B)
            losses_style = loss_style.item()

        # loss percep
        losses_vgg = 0
        if 'Lpercep' in self.config.helper:
            loss_vgg = self.criterions['VGGLoss'](fake_B, real_B)
            losses_vgg = loss_vgg.item()

        # loss boundary
        loss_boundary_coarse = torch.zeros(1).cuda()
        losses_boundary_coarse = 0
        loss_boundary = torch.zeros(1).cuda()
        losses_boundary = 0
        if 'Richer' in self.config.helper:
            # loss coarse boundary
            if boundary_out_coarse is not None:
                for o in boundary_out_coarse:
                    loss_boundary_coarse = loss_boundary_coarse + \
                                           self.criterions['L1LossMaskedMean'](o, real_B, boundary_B)
                losses_boundary_coarse = loss_boundary_coarse.item()

            # loss boundary
            if boundary_out is not None:
                for o in boundary_out:
                    loss_boundary = loss_boundary + self.criterions['L1LossMaskedMean'](o, real_B, boundary_B)
                losses_boundary = loss_boundary.item()

        # loss recon
        losses_tumor_coarse = 0
        losses_tumor = 0
        if 'Lcon' in self.config.helper:
            # loss coarse tumor
            if fake_B_coarse is not None:
                loss_tumor_coarse = self.criterions['L1LossMaskedMean'](fake_B_coarse, real_B, tumor_B)
                losses_tumor_coarse = loss_tumor_coarse.item()

            # loss tumor
            if fake_B is not None:
                loss_tumor = self.criterions['L1LossMaskedMean'](fake_B, real_B, tumor_B)
                losses_tumor = loss_tumor.item()

        # # Pixel-wise loss
        losses_pixel = 0
        if real_B is not None:
            loss_pixel = self.criterions['criterion_pixelwise'](fake_B, real_B)
            losses_pixel = loss_pixel.item()

        losses_pixel_coarse = 0
        if fake_B_coarse is not None:
            loss_pixel_coarse = self.criterions['criterion_pixelwise'](fake_B_coarse, real_B)
            losses_pixel_coarse = loss_pixel_coarse.item()

        return {
            "loss_pixel": losses_pixel,
            "loss_pixel_coarse": losses_pixel_coarse,
            "loss_boundary": losses_boundary,
            "loss_boundary_coarse": losses_boundary_coarse,
            "loss_tumor": losses_tumor,
            "loss_tumor_coarse": losses_tumor_coarse,
            "fake_B": fake_B,
            "fake_B_coarse": fake_B_coarse,
            "boundary": boundary_out,
            "boundary_coarse": boundary_out_coarse,
            "loss_percep": losses_vgg,
            "loss_style": losses_style
        }

    def train_generator_one_batch_pretrained(self, batch_gen):
        real_A = batch_gen['real_A']
        real_B = batch_gen['real_B']
        boundary_B = batch_gen['boundary_B']
        tumor_B = batch_gen['tumor_B']
        # generator output, inpuy and mask
        g_out = self.generator(real_A, tumor_B, pre_trained=True)
        fake_B = g_out['outputs']
        boundary_out = g_out['so_out']
        self.set_requires_grad(self.discriminator, False)

        # loss gan
        pred_fake = self.discriminator(torch.cat((fake_B.detach(), tumor_B), 1))
        out_shape = (pred_fake.size(0), 1, pred_fake.size(2), pred_fake.size(3), pred_fake.size(4))
        valid_label = self.FloatTensor(np.ones(out_shape))
        loss_fake = self.criterions['criterion_GAN'](pred_fake, valid_label)
        loss_GAN_final = loss_fake
        losses_GAN = loss_GAN_final.item()

        # loss style
        loss_style = torch.zeros(1).cuda()
        losses_style = 0
        if 'Lsty' in self.config.helper:
            loss_style = self.criterions['StyleLoss'](fake_B, real_B)
            losses_style = loss_style.item()

        # loss percep
        loss_vgg = torch.zeros(1).cuda()
        losses_vgg = 0
        if 'Lpercep' in self.config.helper:
            loss_vgg = self.criterions['VGGLoss'](fake_B, real_B)
            losses_vgg = loss_vgg.item()

        # loss boundary
        loss_boundary_coarse = torch.zeros(1).cuda()
        losses_boundary_coarse = 0
        loss_boundary = torch.zeros(1).cuda()
        losses_boundary = 0
        if 'Richer' in self.config.helper:
            # loss boundary
            if boundary_out is not None:
                for o in boundary_out:
                    loss_boundary = loss_boundary + self.criterions['L1LossMaskedMean'](o, real_B, boundary_B)
                losses_boundary = loss_boundary.item()

        # loss recon
        loss_tumor_coarse = torch.zeros(1).cuda()
        losses_tumor_coarse = 0
        loss_tumor = torch.zeros(1).cuda()
        losses_tumor = 0
        if 'Lcon' in self.config.helper:
            # loss tumor
            if fake_B is not None:
                loss_tumor = self.criterions['L1LossMaskedMean'](fake_B, real_B, tumor_B)
                losses_tumor = loss_tumor.item()

        # # Pixel-wise loss
        loss_pixel = torch.zeros(1).cuda()
        losses_pixel = 0
        if real_B is not None:
            loss_pixel = self.criterions['criterion_pixelwise'](fake_B, real_B)
            losses_pixel = loss_pixel.item()

        loss_pixel_coarse = torch.zeros(1).cuda()
        losses_pixel_coarse = 0

        # Total loss
        loss_G = loss_GAN_final + self.config.lambda_fix * (
                self.config.percep_loss_factor * loss_vgg +
                self.config.style_loss_factor * loss_style +
                self.config.lambda_pixel * (
                        loss_pixel + loss_pixel_coarse) +
                self.config.tumor_loss_factor * (
                        loss_tumor + loss_tumor_coarse) +
                self.config.boundary_loss_factor * (
                        loss_boundary + loss_boundary_coarse))
        # loss_G = loss_GAN + lambda_pixel * loss_pixel
        losses_G = loss_G.item()
        loss_G.backward()
        return {
            "loss_GAN": losses_GAN,
            "loss_pixel": losses_pixel,
            "loss_pixel_coarse": losses_pixel_coarse,
            "loss_boundary": losses_boundary,
            "loss_boundary_coarse": losses_boundary_coarse,
            "loss_tumor": losses_tumor,
            "loss_tumor_coarse": losses_tumor_coarse,
            "loss_G": losses_G,
            "fake_B": fake_B,
            "loss_percep": losses_vgg,
            "loss_style": losses_style
        }

    def train_generator_one_batch(self, batch_gen):
        real_A = batch_gen['real_A']
        real_B = batch_gen['real_B']
        boundary_B = batch_gen['boundary_B']
        tumor_B = batch_gen['tumor_B']
        # generator output, inpuy and mask
        g_out = self.generator(real_A, tumor_B)
        fake_B = g_out['outputs']
        boundary_out = g_out['so_out']
        fake_B_coarse = g_out['coarse_outputs']
        boundary_out_coarse = g_out['so_out_c']
        self.set_requires_grad(self.discriminator, False)

        # loss gan
        pred_fake = self.discriminator(torch.cat((fake_B.detach(), tumor_B), 1))
        out_shape = (pred_fake.size(0), 1, pred_fake.size(2), pred_fake.size(3), pred_fake.size(4))
        valid_label = self.FloatTensor(np.ones(out_shape))
        loss_fake = self.criterions['criterion_GAN'](pred_fake, valid_label)
        loss_GAN_final = loss_fake
        losses_GAN = loss_GAN_final.item()

        # loss style
        loss_style = torch.zeros(1).cuda()
        losses_style = 0
        if 'Lsty' in self.config.helper:
            loss_style = self.criterions['StyleLoss'](fake_B, real_B)
            losses_style = loss_style.item()

        # loss percep
        loss_vgg = torch.zeros(1).cuda()
        losses_vgg = 0
        if 'Lpercep' in self.config.helper:
            loss_vgg = self.criterions['VGGLoss'](fake_B, real_B)
            losses_vgg = loss_vgg.item()

        # loss boundary
        loss_boundary_coarse = torch.zeros(1).cuda()
        losses_boundary_coarse = 0
        loss_boundary = torch.zeros(1).cuda()
        losses_boundary = 0
        if 'Richer' in self.config.helper:
            # loss coarse boundary
            if boundary_out_coarse is not None:
                for o in boundary_out_coarse:
                    loss_boundary_coarse = loss_boundary_coarse + \
                                           self.criterions['L1LossMaskedMean'](o, real_B, boundary_B)
                losses_boundary_coarse = loss_boundary_coarse.item()

            # loss boundary
            if boundary_out is not None:
                for o in boundary_out:
                    loss_boundary = loss_boundary + self.criterions['L1LossMaskedMean'](o, real_B, boundary_B)
                losses_boundary = loss_boundary.item()

        # loss recon
        loss_tumor_coarse = torch.zeros(1).cuda()
        losses_tumor_coarse = 0
        loss_tumor = torch.zeros(1).cuda()
        losses_tumor = 0
        if 'Lcon' in self.config.helper:
            # loss coarse tumor
            if fake_B_coarse is not None:
                loss_tumor_coarse = self.criterions['L1LossMaskedMean'](fake_B_coarse, real_B, tumor_B)
                losses_tumor_coarse = loss_tumor_coarse.item()

            # loss tumor
            if fake_B is not None:
                loss_tumor = self.criterions['L1LossMaskedMean'](fake_B, real_B, tumor_B)
                losses_tumor = loss_tumor.item()

        # # Pixel-wise loss
        loss_pixel = torch.zeros(1).cuda()
        losses_pixel = 0
        if real_B is not None:
            loss_pixel = self.criterions['criterion_pixelwise'](fake_B, real_B)
            losses_pixel = loss_pixel.item()

        loss_pixel_coarse = torch.zeros(1).cuda()
        losses_pixel_coarse = 0
        if fake_B_coarse is not None:
            loss_pixel_coarse = self.criterions['criterion_pixelwise'](fake_B_coarse, real_B)
            losses_pixel_coarse = loss_pixel_coarse.item()

        # Total loss
        loss_G = loss_GAN_final + self.config.lambda_fix * (
                self.config.percep_loss_factor * loss_vgg +
                self.config.style_loss_factor * loss_style +
                self.config.lambda_pixel * (
                        loss_pixel + loss_pixel_coarse) +
                self.config.tumor_loss_factor * (
                        loss_tumor + loss_tumor_coarse) +
                self.config.boundary_loss_factor * (
                        loss_boundary + loss_boundary_coarse))
        # loss_G = loss_GAN + lambda_pixel * loss_pixel
        losses_G = loss_G.item()
        loss_G.backward()
        return {
            "loss_GAN": losses_GAN,
            "loss_pixel": losses_pixel,
            "loss_pixel_coarse": losses_pixel_coarse,
            "loss_boundary": losses_boundary,
            "loss_boundary_coarse": losses_boundary_coarse,
            "loss_tumor": losses_tumor,
            "loss_tumor_coarse": losses_tumor_coarse,
            "loss_G": losses_G,
            "fake_B": fake_B,
            "loss_percep": losses_vgg,
            "loss_style": losses_style
        }
