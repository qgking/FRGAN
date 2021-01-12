from tumor_data.SYNDataLoader import *
from tensorboardX import SummaryWriter
from torchsummaryX import summary
from driver.helper.base_syn_helper import BaseTrainHelper


class SYNHelper_Dakai(BaseTrainHelper):
    def __init__(self, generator, discriminator,
                 criterions, config):
        super(SYNHelper_Dakai, self).__init__(generator, discriminator,
                                           criterions, config)


    def out_put_shape(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)
        # summary(self.generator.cpu(),
        #         torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)),
        #         torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)))
        # summary(self.discriminator.cpu(),
        #         torch.zeros((1, 2, self.config.patch_x, self.config.patch_y, self.config.patch_z)))


    def test_one_batch(self, batch_gen):
        real_A = batch_gen['real_A']
        real_B = batch_gen['real_B']
        boundary_B = batch_gen['boundary_B']
        tumor_B = batch_gen['tumor_B']

        fake_B = self.generator(real_A, tumor_B)

        loss_boundary = torch.mean(torch.mul(boundary_B, torch.abs(fake_B - real_B)))
        loss_pixel = self.criterions['criterion_pixelwise'](fake_B, real_B)

        losses = loss_pixel.item()
        losses_boundary = loss_boundary.item()
        return {
            "loss_pixel": losses,
            "loss_boundary": losses_boundary,
            "fake_B": fake_B,
            "loss_tumor": 0,
            "loss_style": 0,
            "loss_percep": 0,
            "loss_pixel_coarse": 0,
            "loss_boundary_coarse": 0,
            "fake_B_coarse": None,
            "boundary": None,
            "boundary_coarse": None,
            "loss_tumor_coarse": 0,
        }

    def train_generator_one_batch(self, batch_gen):
        real_A = batch_gen['real_A']
        real_B = batch_gen['real_B']
        boundary_B = batch_gen['boundary_B']
        tumor_B = batch_gen['tumor_B']
        # GAN loss
        fake_B = self.generator(real_A, tumor_B)

        # loss gan
        pred_fake = self.discriminator(torch.cat((fake_B.detach(), tumor_B), 1))
        out_shape = (pred_fake.size(0), 1, pred_fake.size(2), pred_fake.size(3), pred_fake.size(4))
        valid_label = self.FloatTensor(np.ones(out_shape))
        loss_fake = self.criterions['criterion_GAN'](pred_fake, valid_label)
        loss_GAN_final = loss_fake
        losses_GAN = loss_GAN_final.item()

        # # Pixel-wise loss
        loss_pixel = self.criterions['criterion_pixelwise'](fake_B, real_B)
        losses_pixel = loss_pixel.item()
        # # boundary loss
        loss_boundary = torch.mean(torch.mul(boundary_B, torch.abs(fake_B - real_B)))
        losses_boundary = loss_boundary.item()

        # Total loss
        # loss_G = loss_GAN_final + self.config.lambda_fix * (
        #         self.config.lambda_pixel * loss_pixel + self.config.boundary_loss_factor * loss_boundary)
        loss_G = loss_GAN_final + 100 * (
                loss_pixel + 5 * loss_boundary)
        losses_G = loss_G.item()
        loss_G.backward()
        return {
            "loss_GAN": losses_GAN,
            "loss_pixel": losses_pixel,
            "loss_boundary": losses_boundary,
            "loss_tumor": 0,
            "loss_pixel_coarse": 0,
            "loss_boundary_coarse": 0,
            "loss_tumor_coarse": 0,
            "loss_G": losses_G,
            "fake_B": fake_B,
            "loss_percep": 0,
            "loss_style": 0

        }

    # def train_discriminator_one_batch(self, batch_gen):
    #     fake_B = batch_gen['fake_B']
    #     real_B = batch_gen['real_B']
    #     tumor_B = batch_gen['tumor_B']
    #
    #     fake_patch = torch.cat([fake_B, tumor_B], 1)
    #     real_patch = torch.cat([real_B, tumor_B], 1)
    #
    #     pred_fake = self.discriminator(fake_patch.detach())
    #     pred_real = self.discriminator(real_patch)
    #     loss_D = self.criterions['criterion_GAN'](pred_fake, pred_real.detach())
    #
    #     # loss_GAN_g_1 = self.criterion_GAN(pred_global, valid)
    #     # loss_GAN_g_2 = self.criterion_GAN_2(pred_global, valid)
    #     # loss_GAN_final = loss_GAN_g_1 * 0.001 + loss_GAN_g_2 * 0.999
    #     # loss_GAN_final = loss_GAN_final * weight
    #
    #     losses_D = loss_D.item()
    #     loss_D.backward()
    #     return losses_D