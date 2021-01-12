from tumor_data.SYNDataLoader import *
from os.path import abspath, dirname, isdir, join
from utils.utils import *
from tensorboardX import SummaryWriter
from torchsummaryX import summary


class SEGHelper(object):
    def __init__(self, model, critic, config):
        self.model = model
        self.critic = critic
        self.config = config
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

    def save_checkpoint(self, state, filename='checkpoint.pth'):
        torch.save(state, filename)

    def make_dirs(self):
        if not isdir(self.config.tmp_dir):
            os.makedirs(self.config.tmp_dir)
        if not isdir(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        if not isdir(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        if not isdir(self.config.tensorboard_dir):
            os.makedirs(self.config.tensorboard_dir)

    def out_put_shape(self):
        # print(self.model)
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)
        # summary(self.model.cpu(),
        #         torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)))
        # loss = self.critic(torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)),
        #                    torch.zeros((1, 1, self.config.patch_x, self.config.patch_y, self.config.patch_z)))

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
            self.model.cuda()
            self.critic.cuda()

    def generate_batch(self, batch):
        image = batch['img_ori']
        label = batch['img_seg']
        if self.use_cuda:
            image = image.cuda(self.device).float()
            label = label.cuda(self.device).float()
        real_A = self.FloatTensor(image).requires_grad_(False)
        real_B = self.FloatTensor(label).requires_grad_(False)
        return real_A, real_B

    def test_one_batch(self, real_A, real_B):
        fake_B = self.model(real_A)
        loss_pixel = self.critic(fake_B, real_B)
        losses = loss_pixel.item()
        acc = self.accuracy_check(fake_B, real_B)
        return losses, acc, fake_B

    def train_model_one_batch(self, real_A, real_B):
        fake_B = self.model(real_A)
        loss = self.critic(fake_B, real_B)
        losses = loss.item()
        loss.backward()
        acc = self.accuracy_check(fake_B, real_B)
        return losses, acc

    def save_model_checkpoint(self, epoch, optimizer):
        save_file = join(self.config.save_model_path, 'checkpoint_epoch_%03d.pth' % (epoch + 1))
        self.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)

    def write_train_summary(self, epoch, loss):
        self.summary_writer.add_scalar(
            'train/loss', loss, epoch)

    def write_vali_summary(self, epoch, loss):
        self.summary_writer.add_scalar(
            'vali/loss', loss, epoch)

    def adjust_learning_rate(self, optimizer, i_iter, num_steps):
        warmup_iter = num_steps // 20
        if i_iter < warmup_iter:
            lr = self.lr_warmup(self.config.learning_rate, i_iter, warmup_iter)
        else:
            lr = self.lr_poly(self.config.learning_rate, i_iter, num_steps, 0.9)
        optimizer.param_groups[0]['lr'] = lr

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def lr_warmup(self, base_lr, iter, warmup_iter):
        return base_lr * (float(iter) / warmup_iter)

    def accuracy_check(self, mask, prediction):
        ims = [mask, prediction]
        np_ims = []
        for item in ims:
            if 'PIL' in str(type(item)):
                item = np.array(item)
            elif 'torch' in str(type(item)):
                item = item.cpu().detach().numpy()
            np_ims.append(item)
        compare = np.equal(np.where(np_ims[0] > 0.5, 1, 0), np_ims[1])
        accuracy = np.sum(compare)
        return accuracy / len(np_ims[0].flatten())
