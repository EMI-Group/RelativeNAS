from tools.collections import AttrDict

__C = AttrDict()

cfg = __C

__C.train_params = AttrDict()
__C.train_params.epochs = 250
__C.train_params.use_seed = False
__C.train_params.seed = 0

__C.optim = AttrDict()
__C.optim.init_lr = 0.5     # 0.5, 0.1
__C.optim.min_lr = 1e-5
__C.optim.lr_schedule = 'linear'  # cosine poly, linear

__C.optim.momentum = 0.9
__C.optim.weight_decay = 3e-5
__C.optim.use_grad_clip = False
__C.optim.grad_clip = 5

__C.optim.label_smooth = True
__C.optim.smooth_alpha = 0.1

__C.optim.auxiliary = True
__C.optim.auxiliary_weight = 0.4

__C.optim.if_resume = False
__C.optim.resume = AttrDict()
__C.optim.resume.load_path = ''
__C.optim.resume.load_epoch = 0

__C.data = AttrDict()
__C.data.num_workers = 65   # 16, 32
__C.data.batch_size = 1024  # 512, 1024
__C.data.dataset = 'imagenet'
__C.data.train_data_type = 'lmdb'
__C.data.val_data_type = 'lmdb'
__C.data.patch_dataset = False
__C.data.num_examples = 1281167
__C.data.input_size = (3, 224, 224)
__C.data.scaled_size = (3, 256, 256)
__C.data.type_of_data_aug = 'random_sized'  # random_sized / rand_scale
__C.data.random_sized = AttrDict()
__C.data.random_sized.min_scale = 0.08
__C.data.mean = [0.485, 0.456, 0.406]
__C.data.std = [0.229, 0.224, 0.225]
__C.data.color = True