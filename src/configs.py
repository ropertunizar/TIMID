
def build_config(dataset):
    cfg = type('', (), {})()
    if dataset in ['ucf', 'ucf-crime', 'ucf2']:
        cfg.dataset = 'ucf2'
        cfg.model_name = 'ucf_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = './ucf-crime/UCF_ten_onecrop' # './ucf-crime/UCF_ten_onecrop'
        cfg.train_list = './list/ucf2/train.list'
        cfg.test_list = './list/ucf2/test.list'
        cfg.token_feat = './list/ucf2/ucf2-prompt.npy'
        # cfg.gt = './list/ucf/ucf-gt-updated.npy' #'./list/ucf/gt-ucf.npy' # './list/ucf/ucf-gt.npy'
        cfg.gt = './list/ucf2/ucf-gt-updated.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.6
        cfg.bias = 0.2
        cfg.norm = True
        # CC settings
        cfg.t_step = 9
        # training settings
        cfg.temp = 0.09
        cfg.lamda = 1
        cfg.seed = 9
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'slide'  # ['fixed': 10, slide': 7]
        cfg.kappa = 7  # smooth window
    

    elif dataset in ['xd', 'xd-violence']:
        cfg.dataset = 'xd-violence'
        cfg.model_name = 'xd_'
        cfg.metrics = 'AP'
        cfg.feat_prefix = '/data/pyj/feat/xd-i3d'
        cfg.train_list = './list/xd/train.list'
        cfg.test_list = './list/xd/test.list'
        cfg.token_feat = './list/xd/xd-prompt.npy'
        cfg.gt = './list/xd/xd-gt.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.06
        cfg.bias = 0.02
        cfg.norm = False
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.05
        cfg.lamda = 1
        cfg.seed = 4
        # test settings
        cfg.test_bs = 5
        cfg.smooth = 'fixed'  # ['fixed': 8, slide': 3]
        cfg.kappa = 8  # smooth window


    elif dataset in ['sh', 'SHTech']:
        cfg.dataset = 'shanghaiTech'
        cfg.model_name = 'SH_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = '/data/pyj/feat/SHTech-i3d'
        cfg.train_list = './list/sh/train.list'
        cfg.test_list = './list/sh/test.list'
        cfg.token_feat = './list/sh/sh-prompt.npy'
        cfg.abn_label = './list/sh/relabel.list'
        cfg.gt = './list/sh/sh-gt.npy'
        # TCA settings
        cfg.win_size = 5
        cfg.gamma = 0.08
        cfg.bias = 0.1
        cfg.norm = True
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.2
        cfg.lamda = 9
        cfg.seed = 0
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'slide'  # ['fixed': 5, slide': 3]
        cfg.kappa = 3  # smooth window

    elif dataset in ['bridge']:
        cfg.dataset = 'bridge'
        cfg.model_name = 'bridge_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = './bridge/i3d_onecrop'
        cfg.train_list = './list/bridge/train.list'
        cfg.test_list = './list/bridge/test.list'
        cfg.token_feat = './list/bridge/bridge-prompt.npy'
        cfg.gt = './list/bridge/gt.npy'
        # TCA settings
        cfg.win_size = 8
        cfg.gamma = 0.08
        cfg.bias = 0.1
        cfg.norm = True
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.005
        cfg.lamda = 1.2
        cfg.seed = 0
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'None'  # ['fixed': 5, slide': 3]
        cfg.kappa = 3  # smooth window
 
    elif dataset in ['ordering']:
        cfg.dataset = 'ordering'
        cfg.model_name = 'ordering_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = './ordering/i3d_onecrop'
        cfg.train_list = './list/ordering/train_list.list'
        cfg.test_list = './list/ordering/test_list.list'
        cfg.token_feat = './list/ordering/ordering-prompt.npy'
        cfg.gt = './list/ordering/gt.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.08
        cfg.bias = 0.2
        cfg.norm = True
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.005
        cfg.lamda = 1
        cfg.seed = 0
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'None'  # ['fixed': 5, slide': 3]
        cfg.kappa = 3  # smooth window

    elif dataset in ['mutex']:
        cfg.dataset = 'mutex'
        cfg.model_name = 'mutex_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = './mutex/i3d_onecrop'
        cfg.train_list = './list/mutex/train_list.list'
        cfg.test_list = './list/mutex/test_list.list'
        cfg.token_feat = './list/mutex/mutex-prompt.npy'
        cfg.gt = './list/mutex/gt.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.08
        cfg.bias = 0.2
        cfg.norm = True
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.005
        cfg.lamda = 1
        cfg.seed = 0
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'None'  # ['fixed': 5, slide': 3]
        cfg.kappa = 3  # smooth window

    elif dataset in ['ordering_real']:
        cfg.dataset = 'ordering_real'
        cfg.model_name = 'ordering_real_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = './ordering_real/i3d_onecrop'
        cfg.train_list = './list/ordering_real/train_list.list'
        cfg.test_list = './list/ordering_real/test_list.list'
        cfg.token_feat = './list/ordering_real/ordering-prompt.npy'
        cfg.gt = './list/ordering_real/gt.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.08
        cfg.bias = 0.2
        cfg.norm = True
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.05
        cfg.lamda = 1
        cfg.seed = 0
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'None'  # ['fixed': 5, slide': 3]
        cfg.kappa = 3  # smooth window

    elif dataset in ['mutex_real']:
        cfg.dataset = 'mutex_real'
        cfg.model_name = 'mutex_real_'
        cfg.metrics = 'AUC'
        cfg.feat_prefix = './mutex_real/i3d_onecrop'
        cfg.train_list = './list/mutex_real/train_list.list'
        cfg.test_list = './list/mutex_real/test_list.list'
        cfg.token_feat = './list/mutex_real/mutex-prompt.npy'
        cfg.gt = './list/mutex_real/gt.npy'
        # TCA settings
        cfg.win_size = 9
        cfg.gamma = 0.08
        cfg.bias = 0.2
        cfg.norm = True
        # CC settings
        cfg.t_step = 3
        # training settings
        cfg.temp = 0.05
        cfg.lamda = 1
        cfg.seed = 0
        # test settings
        cfg.test_bs = 10
        cfg.smooth = 'None'  # ['fixed': 5, slide': 3]
        cfg.kappa = 3  # smooth window


    # base settings
    cfg.feat_dim = 1024
    cfg.head_num = 1
    cfg.hid_dim = 128
    cfg.out_dim = 300
    cfg.lr = 5e-5
    cfg.dropout = 0.2
    cfg.train_bs = 12
    cfg.max_seqlen = 200
    cfg.max_epoch = 50
    cfg.workers = 8
    cfg.save_dir = './ckpt/'
    cfg.logs_dir = './log_info.log'
    cfg.plot = False

    return cfg
