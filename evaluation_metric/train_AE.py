import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from evaluation_metric.embedding_net import EmbeddingNet
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_iter(target_data, net, optim):
    # zero gradients
    optim.zero_grad()

    # reconstruction loss
    feat, recon_data = net(target_data)
    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    if True:  # use pose diff
        target_diff = target_data[:, 1:] - target_data[:, :-1]
        recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
        recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    recon_loss.backward()
    optim.step()

    ret_dict = {'loss': recon_loss.item()}
    return ret_dict


def make_tensor(path, n_frames, dataset, stride=None, max_files=64, n_chunks=None):
    # use calculate_mean_std.py to get mean/std values
    #mean_vec = np.array([0.06847747184608119,0.0357063580846738,0.058578770784627525,0.06847747184608119,183.03570635799917,0.058578770784627525,16.923291745240235,168.35926808117108,0.041955713046869694,22.215592763704525,96.00786795875209,7.367188758228035,-17.300431170968334,168.94294163004272,-0.24727338642555735,-20.8715950848037,96.38631298614739,5.896949875381165,0.20216133904775593,186.95854383542118,-6.432579379678317,0.6098873474970944,208.91504190095858,-4.915889192590782,1.0909151123302776,229.8609722171563,-12.6793862809497,1.928591364103774,266.5012367723191,-15.976334397752815,7.714763567890464,280.55824733756486,1.9359663647433214,39.544279960636416,270.51838703291963,-9.270711837971861,39.544279960636416,270.51838703291963,-9.270711837971861,50.81093482089116,222.26787089133526,-4.277154500217943,50.17939487042671,194.5012435696503,21.649902584660502,47.55546776177503,193.07937915800173,24.449814516984887,51.68258742769814,183.1130622846742,28.598284340608124,52.31257982349485,178.7300103572332,29.752167196655417,52.11476016985293,176.2234409034844,29.338322795250296,49.57245060363262,183.6685185194845,30.764422876178404,49.51180617225469,177.9781510143326,31.345989263749335,48.984835786867286,174.72433725238744,30.80040989078595,46.938570200832714,184.18559365577823,32.38094931234402,46.63205116111747,177.99271292951627,33.3617908178335,45.70316718692669,175.00854125031574,32.71919596837255,43.53471060814086,184.72586361214883,32.65900465541941,43.144235682959376,179.3784708387911,34.567474274114595,42.55463405258724,176.60663073315192,34.61292128747094,46.18735692733591,192.1666386454192,25.212175451995105,42.415178023338235,190.40999141207104,24.34708837726526,38.476911084063474,186.96434952919546,26.5919414956809,36.13208594764687,183.66904789495973,28.92036860488664,-3.260172112764866,280.83999461595937,1.9124809183428453,-35.07458946647343,272.1779993206412,-10.188934519524462,-35.07458946647343,272.1779993206412,-10.188934519524462,-44.221430642610315,224.12986592661338,-2.1810619636426996,-48.22807185664309,197.1836151601496,21.820415520252148,-46.33444054575024,195.86804954902195,25.039544449623758,-45.13668197604728,194.955477133584,26.116667176956216,-41.092202689799706,193.1729063777209,26.266446799572222,-37.40708496857298,189.6782655958083,29.307581254228797,-35.636081168567486,186.31843980348182,31.96766893075243,-50.79671007842462,186.06135889526658,28.08238050295619,-51.43370660186226,181.74414430313067,29.05281490814699,-50.935496917944754,179.24853570625916,28.80991130269415,-47.240733921717286,187.10254399331873,32.83320587793653,-46.90782281703879,180.84945866262322,34.0909044604069,-45.500171172703396,177.78021908057028,33.90137244712666,-49.34991372818972,186.60529710991787,30.652207359792364,-49.09096325437287,180.95734931633348,31.311876315818232,-48.05107944797273,177.7834688982274,31.007569225066618,-44.02123442944895,187.60657246793522,33.93921950361445,-43.76597286425863,182.3069165226035,36.05010015011614,-42.954543530752,179.48150399466223,36.4037713410187,2.5187557703509666,291.83247385777486,-9.389368690830807,2.4539357753355553,304.03709615227046,-2.881607696619349])
    #std_vec = np.array([8.255541100752382,0.6939720601761805,5.222482738827612,8.255541100752382,0.693972060176339,5.222482738827612,9.42100963255725,2.188372644619285,6.361481612445378,6.176403474432683,2.03443820178515,11.830267060993494,9.348297312453099,1.9986317079084224,6.001779897668299,8.162462099200143,2.2708856073173265,12.143746363244206,8.07002281773558,1.146760813248187,5.218745502524028,7.302334393532628,1.0869661835365756,5.640455884072193,7.477263866275783,1.2241260130341458,5.704034599390462,8.93140892031836,1.3043997705897612,6.246232138662631,9.656330236153833,1.7514046003701789,6.8102931179285715,9.369187221685337,3.7379417578069427,9.746046092317828,9.369187221685337,3.7379417578069427,9.746046092317828,10.227838670518729,7.286263937298487,13.107272130744121,17.056602336480278,28.775355928036703,17.65156414660871,17.616759573364362,31.57948329721122,17.58583620094351,22.409700875985976,36.64470600046972,22.894290518164947,24.53008313723881,39.07577209642486,25.295457925596633,25.690655337228268,40.177065667200765,26.808543783812727,22.2591494319307,38.401747944200125,21.950631662738534,24.854161656529712,41.50067854859389,25.042011259666936,26.473295311534336,43.197994006554005,27.221170133903826,22.10861269363438,40.10343062678119,21.14882281947506,25.013596742327397,43.87399009904381,24.52999507148788,26.52013512923851,45.59851255330215,26.774862236150504,21.725195479879996,41.3712584397316,20.536614369930735,24.39943341761268,45.32943762847826,23.38680984964916,25.786178720369854,47.14171501427746,25.319483052572473,17.914690428682025,32.89781536653379,17.7403567588838,18.181332875330988,34.64247649156103,18.418610850063928,19.771894506185426,39.40693746620956,19.868089428475518,21.55081438729135,43.62997102172089,21.5116223452909,9.702982004349828,1.7826399048001014,6.831726011211061,9.990627742798965,4.4897264818483995,9.924705741596368,9.990627742798965,4.4897264818483995,9.924705741596368,11.786532337428321,9.522123886195812,12.65776670636571,17.031859201169226,33.584280601824254,18.38674479122489,17.777049225781234,36.38401573078446,18.127686404978764,18.17511401763394,37.691189634409334,18.084351816160314,18.551140073870705,39.31562974080336,18.02247544203335,20.28661077866561,43.99125147305048,18.701801984691905,22.30030442409078,48.11743158915539,19.99370126076239,21.89724529987023,42.49461135019713,23.793704882062592,23.74127743788964,45.32675754902733,26.24880800837308,24.594991051498763,46.72175188102939,27.67702463406083,22.183607917647112,45.59612353311565,21.613578453862164,24.6049569241885,50.12226277439626,24.67927794755758,25.732481639988514,52.301113166777355,26.560623330610035,22.023636510505785,44.0923860001992,22.687839346541782,24.11276937550739,47.8236277337321,25.586912334755322,25.285414927567018,49.8865610793381,27.509628579848478,22.140064803278623,46.596465179326735,20.51837959275402,24.421617248855064,51.09661706141259,22.99069540500737,25.553481257475042,53.30714083193669,24.588120394659704,10.509501956123042,1.3845656231709993,7.232725520260621,10.735209001060486,1.9048225062899646,7.924571615243941])
    if dataset == 'genea':
        mean_vec = np.load('./dataset/Genea2023/trn/main-agent/rotpos_Mean.npy')
        std_vec = np.load('./dataset/Genea2023/trn/main-agent/rotpos_Std.npy')
        idx_positions = np.asarray([ [i*6+3, i*6+4, i*6+5] for i in range(int(len(mean_vec)/6)) ]).flatten()
        mean_vec = mean_vec[idx_positions]
        std_vec = std_vec[idx_positions]
    elif dataset == 'ptbr':
        std_vec = np.load('./dataset/PTBRGestures/pos_Std.npy')
        mean_vec = np.load('./dataset/PTBRGestures/pos_Mean.npy')
        idx_positions = np.arange(len(mean_vec))
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    
    std_vec[std_vec==0] = 1
    
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.npy'))
    else:
        raise ValueError('Unknown path: {}'.format(path))

    return files_to_tensor(files, mean_vec, std_vec, idx_positions=idx_positions, n_frames=n_frames, n_chunks=n_chunks, stride=stride, max_files=max_files)

def files_to_tensor(files, mean_vec, std_vec, idx_positions=None, n_frames=120, n_chunks=None, stride=None, max_files=None):
    # Make sure we don't run out of memory
    max_files = max_files if max_files < len(files) else len(files)
    idx_positions = np.arange(len(mean_vec)) if idx_positions is None else idx_positions
    samples = []
    stride = n_frames // 2 if stride is None else stride
    print('Preparing data...')
    for file in files[:max_files]:
        print('Loading {}'.format(file))
        data = np.load(file) #Should be shape [frames, features (joint rotations)]
        data = data[:, idx_positions]
        for i in range(0, len(data) - n_frames, stride)[:n_chunks]:
            sample = data[i:i+n_frames]
            sample = (sample - mean_vec) / std_vec
            samples.append(sample)

    print('Converting to tensor...')
    return torch.Tensor(samples)


def main(n_frames, dataset='genea'):
    #https://github.com/genea-workshop/genea_challenge_2023/tree/main/evaluation_metric
    # dataset
    if dataset == 'genea':
        motion_source = './dataset/Genea2023/trn/main-agent/motion_npy_rotpos'
        max_files = 372
        stride = 60
    elif dataset == 'ptbr':
        motion_source = './dataset/PTBRGestures/motion/pos'
        max_files = 716
        stride = 30
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    train_dataset = TensorDataset(make_tensor(motion_source, n_frames, dataset=dataset, max_files=max_files, stride=stride))
    print('Done')
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)

    # train
    loss_meters = [AverageMeter('loss')]

    # interval params
    print_interval = int(len(train_loader) / 5)

    # init model and optimizer
    pose_dim = 249
    generator = EmbeddingNet(pose_dim, n_frames).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))

    print('Training...')
    # training
    for epoch in range(100):
        for iter_idx, target in enumerate(train_loader, 0):
            target = target[0]
            batch_size = target.size(0)
            target_vec = target.to(device)
            loss = train_iter(target_vec, generator, gen_optimizer)

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | '.format(epoch, iter_idx + 1)
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                print(print_summary)

    # save model
    gen_state_dict = generator.state_dict()
    save_name = f'./evaluation_metric/output/model_checkpoint_{n_frames}_{dataset}.bin'
    torch.save({'pose_dim': pose_dim, 'n_frames': n_frames, 'gen_dict': gen_state_dict}, save_name)


if __name__ == '__main__':
    n_frames = 120
    print('Using n_frames: {}'.format(n_frames))
    main(n_frames, dataset='ptbr')
