from configparser import ConfigParser
import configparser
import sys, os

sys.path.append('..')


class Configurable(object):
    def __init__(self, args, extra_args):
        config = ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(args.config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        config.set('Run', 'gpu', args.gpu)
        config.set('Run', 'gpu_count', args.gpu_count)
        if args.train:
            config.set('Run', 'run_num', args.run_num)
            config.set('Run', 'train_batch_size', args.batch_size)
            config.set('Network', 'helper', args.helper)
            config.set('Network', 'generator', args.gen)
            config.set('Network', 'discriminator', args.dis)
            config.set('Data', 'split_radio', args.split)
            self._config = config
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            config.write(open(self.config_file, 'w'))
        print('Loaded config file sucessfully.')
        self._config = config
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    def set_attr(self, section, name, value):
        return self._config.set(section, name, value)

    # ------------data config reader--------------------
    @property
    def patch_x(self):
        return self._config.getint('Data', 'patch_x')

    @property
    def patch_y(self):
        return self._config.getint('Data', 'patch_y')

    @property
    def patch_z(self):
        return self._config.getint('Data', 'patch_z')

    @property
    def split_radio(self):
        return self._config.getfloat('Data', 'split_radio')

    @property
    def data_name(self):
        return self._config.get('Data', 'data_name')

    @property
    def data_root(self):
        return self._config.get('Data', 'data_root')

    # ------------save path config reader--------------------

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def tmp_dir(self):
        return self._config.get('Save', 'tmp_dir')

    @property
    def tensorboard_dir(self):
        return self._config.get('Save', 'tensorboard_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def log_file(self):
        return self._config.get('Save', 'log_file')

    @property
    def eve_path(self):
        return self._config.get('Save', 'eve_path')

    # ------------Network path config reader--------------------

    @property
    def generator(self):
        return self._config.get('Network', 'generator')

    @property
    def discriminator(self):
        return self._config.get('Network', 'discriminator')

    @property
    def helper(self):
        return self._config.get('Network', 'helper')

    @property
    def seg_net(self):
        return self._config.get('Network', 'seg_net')

    # ------------Network path config reader--------------------

    @property
    def epochs(self):
        return self._config.getint('Run', 'N_epochs')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def gpu(self):
        return self._config.getint('Run', 'gpu')

    @property
    def printfreq(self):
        return self._config.getint('Run', 'printfreq')

    @property
    def lambda_pixel(self):
        return self._config.getfloat('Run', 'lambda_pixel')

    @property
    def lambda_fix(self):
        return self._config.getfloat('Run', 'lambda')

    @property
    def tumor_loss_factor(self):
        return self._config.getfloat('Run', 'tumor_loss_factor')

    @property
    def boundary_loss_factor(self):
        return self._config.getfloat('Run', 'boundary_loss_factor')

    @property
    def percep_loss_factor(self):
        return self._config.getfloat('Run', 'percep_loss_factor')

    @property
    def style_loss_factor(self):
        return self._config.getfloat('Run', 'style_loss_factor')

    @property
    def gpu_count(self):
        gpus = self._config.get('Run', 'gpu_count')
        gpus = gpus.split(',')
        return [int(x) for x in gpus]

    @property
    def workers(self):
        return self._config.getint('Run', 'workers')

    @property
    def run_num(self):
        return self._config.getint('Run', 'run_num')

    @property
    def validate_every_epoch(self):
        return self._config.getint('Run', 'validate_every_epoch')

    @property
    def max_boundary_loss_factor_epoch(self):
        return self._config.getint('Run', 'max_boundary_loss_factor_epoch')

    # ------------Optimizer path config reader--------------------
    @property
    def learning_algorithm(self):
        return self._config.get('Optimizer', 'learning_algorithm')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def start_decay_at(self):
        return self._config.getint('Optimizer', 'start_decay_at')

    @property
    def max_patience(self):
        return self._config.getint('Optimizer', 'max_patience')

    @property
    def min_lrate(self):
        return self._config.getfloat('Optimizer', 'min_lrate')

    @property
    def beta_1(self):
        return self._config.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def decay(self):
        return self._config.getfloat('Optimizer', 'decay')
