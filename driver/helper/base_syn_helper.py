# -*- coding: utf-8 -*-
# @Time    : 20/1/2 21:43
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : base_syn_helper.py
from tumor_data.SYNDataLoader import *
from os.path import abspath, dirname, isdir, join
from utils.utils import *
from tensorboardX import SummaryWriter
from torchsummaryX import summary
from torch import nn
from torchviz import make_dot
from utils.utils import cal_psnr, cal_ssim
from tumor_data.SYNDataLoader import *
from os.path import abspath, dirname, isdir, join
from utils.utils import *
from tensorboardX import SummaryWriter
from torchsummaryX import summary
from torch import nn
from torchviz import make_dot
from utils.utils import cal_psnr, cal_ssim
import matplotlib.pyplot as plt


class BaseTrainHelper(object):
    def __init__(self, generator, discriminator,
                 criterions, config):
        self.generator = generator
        self.discriminator = discriminator
        self.criterions = criterions
        self.config = config
        self.boundary_loss_factor = 0
        # p = next(filter(lambda p: p.requires_grad, generator.parameters()))
        self.use_cuda = config.use_cuda
        # self.device = p.get_device() if self.use_cuda else None
        self.device = config.gpu if self.use_cuda else None
        if self.config.train:
            self.make_dirs()
        self.define_log()
        self.out_put_shape()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

    def make_dirs(self):
        if not isdir(self.config.tmp_dir):
            os.makedirs(self.config.tmp_dir)
        if not isdir(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        if not isdir(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        if not isdir(self.config.tensorboard_dir):
            os.makedirs(self.config.tensorboard_dir)

    def define_log(self):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        if self.config.train:
            log_s = self.config.log_file[:self.config.log_file.rfind('.txt')]
            self.log = Logger(log_s + '_' + str(date_time) + '.txt')
        else:
            self.log = Logger(join(self.config.save_dir, 'test_log_%s.txt' % (str(date_time))))
        sys.stdout = self.log

    def move_to_cuda(self):
        if self.use_cuda:
            torch.cuda.set_device(self.device)
            self.generator.cuda()
            if self.discriminator is not None:
                self.discriminator.cuda()
            for key in self.criterions.keys():
                # d.iteritems: an iterator over the (key, value) items
                # print(key)
                self.criterions[key].cuda()

            if len(self.config.gpu_count) > 1:
                print("GPUs:", self.config.gpu_count)
                self.generator = nn.DataParallel(self.generator, device_ids=self.config.gpu_count)
                self.discriminator = nn.DataParallel(self.discriminator, device_ids=self.config.gpu_count)

    def generate_batch(self, batch):
        image = batch['img_erase']
        label = batch['img_ori']
        tumor_region = batch['tumor_region']
        boundary_region = batch['boundary_region']
        if self.use_cuda:
            image = image.cuda(self.device).float()
            label = label.cuda(self.device).float()
            tumor_region = tumor_region.cuda(self.device).float()
            boundary_region = boundary_region.cuda(self.device).float()
        real_A = self.FloatTensor(image).requires_grad_(False)
        real_B = self.FloatTensor(label).requires_grad_(False)
        tumor_B = self.FloatTensor(tumor_region).requires_grad_(False)
        boundary_B = self.FloatTensor(boundary_region).requires_grad_(False)
        # Adversarial ground truths
        return {
            'real_A': real_A,
            'real_B': real_B,
            'boundary_B': boundary_B,
            'tumor_B': tumor_B,
        }

    # new
    def train_discriminator_one_batch(self, batch_gen):
        self.set_requires_grad(self.discriminator, True)
        fake_B = batch_gen['fake_B']
        real_B = batch_gen['real_B']
        tumor_B = batch_gen['tumor_B']
        pred_fake = self.discriminator(torch.cat((fake_B.detach(), tumor_B), 1))
        pred_real = self.discriminator(torch.cat((real_B, tumor_B), 1))
        out_shape = (pred_fake.size(0), 1, pred_fake.size(2), pred_fake.size(3), pred_fake.size(4))
        valid_label = self.FloatTensor(np.ones(out_shape))
        fake_unlabel = self.FloatTensor(np.zeros(out_shape))
        loss_adv_fake = self.criterions['criterion_GAN'](pred_fake, fake_unlabel)
        loss_adv_real = self.criterions['criterion_GAN'](pred_real, valid_label)
        loss_D = 0.5 * (loss_adv_fake + loss_adv_real)
        losses_D = loss_D.item()
        loss_D.backward()
        return losses_D

    def save_model_checkpoint(self, epoch, model_optimizer):
        old_save_file = join(self.config.save_model_path, 'checkpoint_epoch_%03d.pth' % (epoch))
        if os.path.exists(old_save_file):
            os.remove(old_save_file)
        else:
            print("The file %s does not exist" % (old_save_file))

        save_file = join(self.config.save_model_path, 'checkpoint_epoch_%03d.pth' % (epoch + 1))
        state = {'epoch': epoch,
                 'g_state_dict': self.generator.state_dict(),
                 'd_state_dict': self.discriminator.state_dict(),
                 'g_optimizer': model_optimizer['G'].state_dict(),
                 'd_optimizer': model_optimizer['D'].state_dict()}
        torch.save(state, save_file)

    def load_pretrained_coarse_weight(self):
        weight_files = sorted(glob(join('../log', self.config.data_name, 'checkpoint_epoch_*.pth')), reverse=True)
        if len(weight_files) > 0:
            from collections import OrderedDict
            weight_file = weight_files[0]
            print("loaded history pretrained coarse net weight: %s" % (weight_file))
            net_dict = self.generator.state_dict()
            if not self.config.use_cuda:
                state_dict = torch.load(weight_file,
                                        map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                      'cuda:2': 'cpu', 'cuda:3': 'cpu'})
                net_dict.update(state_dict['g_state_dict'])
                self.generator.load_state_dict(net_dict)
            else:
                state_dict = torch.load(weight_file, map_location=('cuda:' + str(self.device)))
                if 'module' not in list(state_dict['g_state_dict'].keys())[0][:7]:
                    self.generator.load_state_dict(state_dict['g_state_dict'])
                else:
                    if len(self.config.gpu_count) > 1:
                        net_dict.update(state_dict['g_state_dict'])
                        self.generator.load_state_dict(net_dict)
                    else:
                        new_state_dict = OrderedDict()
                        for k, v in state_dict['g_state_dict'].items():
                            name = k[7:]  # remove `module.`
                            new_state_dict[name] = v
                        net_dict.update(new_state_dict)
                        self.generator.load_state_dict(net_dict)
                        del new_state_dict
            del state_dict

    def load_lastest_weight(self, model_optimizer):
        weight_files = sorted(glob(join(self.config.load_model_path, 'checkpoint_epoch_*.pth')), reverse=True)
        if len(weight_files) > 0:
            from collections import OrderedDict
            weight_file = weight_files[0]
            print("loaded history weight: %s" % (weight_file))
            net_dict = self.generator.state_dict()
            if not self.config.use_cuda:
                state_dict = torch.load(weight_file,
                                        map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                      'cuda:2': 'cpu', 'cuda:3': 'cpu'})
                net_dict.update(state_dict['g_state_dict'])
                self.generator.load_state_dict(net_dict)
                self.discriminator.load_state_dict(state_dict['d_state_dict'])
                model_optimizer['G'].load_state_dict(state_dict['g_optimizer'])
                model_optimizer['D'].load_state_dict(state_dict['d_optimizer'])
                epo = state_dict['epoch']
            else:
                state_dict = torch.load(weight_file, map_location=('cuda:' + str(self.device)))
                epo = state_dict['epoch']
                if 'module' not in list(state_dict['g_state_dict'].keys())[0][:7]:
                    net_dict.update(state_dict['g_state_dict'])
                    self.generator.load_state_dict(net_dict)
                    self.discriminator.load_state_dict(state_dict['d_state_dict'])
                else:
                    if len(self.config.gpu_count) > 1:
                        net_dict.update(state_dict['g_state_dict'])
                        self.generator.load_state_dict(net_dict)
                        net_dict = self.discriminator.state_dict()
                        net_dict.update(state_dict['d_state_dict'])
                        self.discriminator.load_state_dict(net_dict)
                    else:
                        new_state_dict = OrderedDict()
                        for k, v in state_dict['g_state_dict'].items():
                            name = k[7:]  # remove `module.`
                            new_state_dict[name] = v
                        net_dict.update(new_state_dict)
                        self.generator.load_state_dict(net_dict)
                        new_state_dict = OrderedDict()
                        for k, v in state_dict['d_state_dict'].items():
                            name = k[7:]  # remove `module.`
                            new_state_dict[name] = v
                        net_dict = self.discriminator.state_dict()
                        net_dict.update(new_state_dict)
                        self.discriminator.load_state_dict(net_dict)
                        del new_state_dict
                model_optimizer['G'].load_state_dict(state_dict['g_optimizer'])
                model_optimizer['D'].load_state_dict(state_dict['d_optimizer'])
            del state_dict
            return epo
        else:
            return 0

    def load_generator_history_weight(self, weight_file):
        if not self.config.use_cuda:
            self.generator.load_state_dict(torch.load(weight_file,
                                                      map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                                    'cuda:2': 'cpu', 'cuda:3': 'cpu'})['g_state_dict'])
        else:
            net_dict = self.generator.state_dict()
            from collections import OrderedDict
            state_dict = torch.load(weight_file, map_location=('cuda:' + str(self.device)))['g_state_dict']
            new_state_dict = OrderedDict()
            if 'module' not in list(state_dict.keys())[0][:7]:
                self.generator.load_state_dict(
                    torch.load(weight_file, map_location=('cuda:' + str(self.device)))['g_state_dict'])
            else:
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                net_dict.update(new_state_dict)
                self.generator.load_state_dict(net_dict)

    def write_summary(self, epoch, criterions):
        for key in criterions.keys():
            self.summary_writer.add_scalar(
                key, criterions[key], epoch)

    def plot_vali_loss(self, epoch, criterions, type='vali'):
        if not hasattr(self, 'content_loss_cal'):
            self.content_loss_cal = []
            self.tumor_loss_cal = []
            self.boundary_loss_cal = []
            self.percp_loss_cal = []
            self.style_loss_cal = []

        plt.figure(figsize=(16, 10), dpi=100)
        self.content_loss_cal.append(criterions[type + '/content_loss'])
        self.tumor_loss_cal.append(criterions[type + '/tumor_loss'])
        self.boundary_loss_cal.append(criterions[type + '/boundary_loss'])
        self.percp_loss_cal.append(criterions[type + '/percp_loss'])
        self.style_loss_cal.append(criterions[type + '/style_loss'])
        epochs = range(len(self.content_loss_cal))
        plt.subplot(1, 1, 1)
        plt.plot(epochs, self.content_loss_cal, color='red', marker='o', linestyle='-', label='content loss')
        plt.plot(epochs, self.tumor_loss_cal, color='blue', marker='s', linestyle='-', label='tumor loss')
        plt.plot(epochs, self.boundary_loss_cal, color='black', marker='v', linestyle='-', label='boundary loss')
        plt.plot(epochs, self.percp_loss_cal, color='yellow', marker='x', linestyle='-', label='percp loss')
        plt.plot(epochs, self.style_loss_cal, color='darkorange', marker='d', linestyle='-', label='style loss')
        plt.ylim(0, 0.5)
        plt.xlim(-2, self.config.epochs + 5)
        plt.title(type + ' loss vs. epoches')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(join(self.config.tmp_dir, type + "_loss.jpg"))
        plt.close()

    def plot_train_loss(self, epoch, criterions, type='train'):
        if not hasattr(self, 'train_loss_D_cal'):
            self.train_loss_D_cal = []
            self.train_loss_G_cal = []
            self.train_loss_GAN_cal = []
            self.train_loss_pixel_cal = []
            self.train_loss_tumor_cal = []
            self.train_loss_boundary_cal = []
            self.train_loss_style_cal = []
            self.train_loss_percep_cal = []

        plt.figure(figsize=(16, 10), dpi=100)
        self.train_loss_D_cal.append(criterions[type + '/loss_D'])
        self.train_loss_G_cal.append(criterions[type + '/loss_G'])
        self.train_loss_GAN_cal.append(criterions[type + '/loss_GAN'])
        self.train_loss_pixel_cal.append(criterions[type + '/loss_pixel'])
        self.train_loss_tumor_cal.append(criterions[type + '/loss_tumor'])
        self.train_loss_boundary_cal.append(criterions[type + '/loss_boundary'])
        self.train_loss_style_cal.append(criterions[type + '/loss_style'])
        self.train_loss_percep_cal.append(criterions[type + '/loss_percep'])

        epochs = range(len(self.train_loss_G_cal))
        plt.subplot(1, 1, 1)
        plt.plot(epochs, self.train_loss_D_cal, color='red', marker='o', linestyle='-', label='D loss')
        plt.plot(epochs, self.train_loss_G_cal, color='blue', marker='s', linestyle='-', label='G loss')
        plt.plot(epochs, self.train_loss_GAN_cal, color='black', marker='v', linestyle='-', label='GAN loss')
        plt.plot(epochs, self.train_loss_pixel_cal, color='yellow', marker='x', linestyle='-', label='Pixel loss')
        plt.plot(epochs, self.train_loss_tumor_cal, color='darkorange', marker='d', linestyle='-', label='Tumor loss')
        plt.plot(epochs, self.train_loss_boundary_cal, color='gold', marker='D', linestyle='-', label='Boundary loss')
        plt.plot(epochs, self.train_loss_style_cal, color='brown', marker='h', linestyle='-', label='Style loss')
        plt.plot(epochs, self.train_loss_percep_cal, color='coral', marker='p', linestyle='-', label='Percep loss')
        plt.ylim(0, 0.3)
        plt.xlim(-2, self.config.epochs + 5)
        plt.title(type + ' loss vs. epoches')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(join(self.config.tmp_dir, type + "_loss.jpg"))
        plt.close()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
