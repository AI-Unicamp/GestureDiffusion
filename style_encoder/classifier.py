import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import bvhsdk
import numpy as np
from style_encoder.encoder_net import StyleClassifier, StyleVAE
from style_encoder.encoder_net import compute_KL_div as KLLoss
from style_encoder.encoder_net import compute_mse as MSE
from data_loaders.gesture.data.ptbrdataset import PTBRGesture
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from style_encoder.parser_util import train_classifier_args
from style_encoder.logger import Logger
import json

class StyleClassifier_PTBRGestures(PTBRGesture):
    def __init__(self, 
                 name,
                 split,
                 step,
                 window,
                 fps=30,
                 datapath='./dataset/PTBRGestures',
                 sr=22050,
                 n_seed_poses=10,
                 use_wavlm=False,
                 use_vad=False,
                 vadfromtext=False):
        super().__init__(name, split, datapath, step, window, fps, sr, n_seed_poses, use_wavlm, use_vad, vadfromtext)
       
    def __getitem__(self, index):
        # find the file that the sample belongs two
        file_idx = np.searchsorted(self.samples_cumulative, index+1, side='left')
        # find sample's index in the file
        sample = index - self.samples_cumulative[file_idx-1] if file_idx > 0 else index
        motion, _ = self._getmotion(file_idx, sample)
        label = self.takes[file_idx].class_label -1
        return motion, label, self.takes[file_idx].name
    


def main():
    args = train_classifier_args()

    if not os.path.exists(args.datapath):
        print(args.path)
        raise ValueError("Datapath {} does not exist".format(args.datapath))
    
    run_name = 'classifier_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataname, 
                                                         args.window_length,
                                                         args.val_step_length,
                                                         args.criterion, 
                                                         args.optimizer, 
                                                         args.hidden_size, 
                                                         args.style_embedding_size, 
                                                         args.learning_rate)
    run_path = os.path.join(args.output_path, run_name)
    if os.path.exists(run_path):
        raise ValueError("Output path {} already exists, can't overwrite".format(run_path))
    else:
        os.makedirs(run_path)

    network, trainloader, valloader, device, criterion, optimizer = prep_data_and_model(args)

    # Save args.json
    with open(os.path.join(run_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    trnlog, vallog = train(path=run_path, 
                            run_name=run_name, 
                            network=network, 
                            trnloader=trainloader, 
                            valloader=valloader, 
                            device=device, 
                            epochs=args.epochs, 
                            optimizer=optimizer, 
                            criterion=criterion,
                            style_encoder=network.styleenc,
                            )

    # Save logs
    trnlog.save(os.path.join(run_path, 'trnlog.json'))
    vallog.save(os.path.join(run_path, 'vallog.json'))



def prep_data_and_model(args, val_only=False):

    if args.dataname == 'ptbr':
        num_classes = 12
        input_size = 1245 # rot6d + pos + vel_pos + vel_rot (6 * 83 + 3 * 83 + 3 * 83 + 3 * 83)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.device == 'gpu') else "cpu")

    if not val_only:
        print('Loading train dataset...')
        trndataset = StyleClassifier_PTBRGestures(name=args.dataname, split='trn', step=args.trn_step_length, window=args.window_length)
        trainloader = DataLoader(trndataset, batch_size=args.batch_size, shuffle=True)

    print('Loading validation dataset...')
    valdataset = StyleClassifier_PTBRGestures(name=args.dataname, split='val', step=args.val_step_length, window=args.window_length)
    valloader   = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False)

    network = StyleClassifier(input_size=input_size, hidden_size=args.hidden_size, style_embedding_size=args.style_embedding_size, out_classes=num_classes).to(device)

    if args.criterion == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion {} not implemented".format(args.criterion))
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum)
    else:
        raise ValueError("Optimizer {} not implemented".format(args.optimizer))
    
    if not val_only:
        return network, trainloader, valloader, device, criterion, optimizer
    else:
        return network, valloader, device
    


def train(path, run_name, network, trnloader, valloader, device, epochs, optimizer, criterion, style_encoder=None):
    trnlog = Logger('logtrn_'+run_name)
    vallog = Logger('logval_'+run_name)
    for epoch in range(epochs):
        network.train()
        trnloss, trnacc = run_epoch(network, trnloader, device, criterion, optimizer)
        trnlog.logbatch(loss=trnloss, acc=trnacc)

        network.eval()
        with torch.no_grad():
            valloss, valacc = run_epoch(network, valloader, device, criterion)
            vallog.logbatch(loss=valloss, acc=valacc)
        
        print("Epoch {} - Trn Loss: {} - Val Loss: {} - Trn Acc: {} - Val Acc: {}".format(epoch, trnloss, valloss, trnacc, valacc))
        if valloss == vallog.min_loss:
            torch.save(network.state_dict(), os.path.join(path, run_name + '.pth'))
            print("Saved model with val loss {}".format(valloss))

            if style_encoder:
                embeddings, labels, samples = run_samples(network, valloader, device)
                fig = run_PCA(embeddings, labels, n_components=2, samples_names=samples)
                fig.savefig(os.path.join(path, run_name + '_epoch_{}.png'.format(epoch)))
                plt.close(fig)
    return trnlog, vallog


def run_epoch(network, loader, device, criterion, optimizer=None, scheduler=None):
    epoch_loss, correct, total = 0.0, 0, 0
    mode = "Training" if optimizer else "Validation"
    with tqdm(loader, desc=f"{mode} Epoch") as pbar:
        for i, batch_data in enumerate(pbar):
            inputs, labels, _ = batch_data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            outputs = network(inputs)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data,1)
            correct += (predicted == labels).sum().item()
            total += predicted.size(0)

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            epoch_loss += loss.item()

            pbar.set_postfix({'Loss': epoch_loss / (i + 1)})
    if scheduler:
        scheduler.step()
    return epoch_loss / len(loader), correct / total

def run_samples(network, loader, device):
    """
    Similar to run_epoch but here we want to get the embeddings for all samples
    """
    network.eval()
    with torch.no_grad():
        embeddings, original_labels, samples = [], [], []
        for j, data in enumerate(loader):
            inputs, labels, name = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            embeddings.extend(network(inputs).detach().cpu().numpy())
            original_labels.extend(labels.detach().cpu().numpy())
            samples.extend(name)
        embeddings = np.array(embeddings)
        original_labels = np.array(original_labels)
        return embeddings, original_labels, samples
    
def run_samples2(network, loader, device):
    """
    Similar to run_epoch but here we want to get the embeddings for all samples.
    Adapted to work for the FGD network
    """
    network.eval()
    with torch.no_grad():
        embeddings, original_labels, samples = [], [], []
        for j, data in enumerate(loader):
            inputs, labels, name = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            f, _ = network(inputs)
            embeddings.extend(f.data.cpu().numpy())
            original_labels.extend(labels.detach().cpu().numpy())
            samples.extend(name)
        embeddings = np.array(embeddings)
        original_labels = np.array(original_labels)
        return embeddings, original_labels, samples

def run_PCA(embbedings, 
            labels,
            n_components, 
            samples_names, 
            x_comp=1, 
            y_comp=2,
            print_extremes=True,
            xlim=None,
            ylim=None,
            title=None):
    """
    x_comp: PCA component to be plotted in the x-axis
    y_comp: PCA component to be plotted in the y-axis
    """
    assert x_comp > 0 and x_comp <= n_components and x_comp != y_comp and y_comp > 0 and y_comp <= n_components
    assert n_components >= 2
    
    pca = PCA(n_components=n_components, random_state=2).fit_transform(embbedings)
    
    x_comp -= 1
    y_comp -= 1

    pca = pca[..., [x_comp, y_comp]]
    
    class_names = ['01extro', '01intro', '01neutral', 
                   '01jov', '01welc', '01formal', 
                   '02extro', '02intro', '02neutral',
                  '02jov', '02welc', '02formal' ]
    class_labels = ['darkred', 'red', 'tomato', 
                    'darkgreen', 'limegreen', 'lime', 
                    'sienna', 'chocolate', 'sandybrown', 
                    'darkblue', 'blue', 'cyan']

    fig = plt.figure(figsize=(12,8))
    ax = plt.gca()
    
    xmin = np.argmin(pca[:,0])
    ymin = np.argmin(pca[:,1])
    xmax = np.argmax(pca[:,0])
    ymax = np.argmax(pca[:,1])
    
    if print_extremes:
        print('-X sample: {} [{},{}] '.format(samples_names[xmin], pca[xmin,0],pca[xmin,1]))
        print('X sample: {} [{},{}] '.format(samples_names[xmax], pca[xmax,0],pca[xmax,1]))
        print('-Y sample: {} [{},{}] '.format(samples_names[ymin], pca[ymin,0],pca[ymin,1]))
        print('Y sample: {} [{},{}] '.format(samples_names[ymax], pca[ymax,0],pca[ymax,1]))
    
    for i in range(12):
        x = pca[labels==i, 0]
        y = pca[labels==i, 1]
        marker = "s" if "01" in class_names[i] else "o"
        scatter = ax.scatter(x, y, c=class_labels[i], label=class_names[i], alpha=0.5, s=20, marker=marker)
        
    #scatter = ax.scatter(pca[:, 0], pca[:, 1], c=[class_labels[color] for color in labels], label=[class_names[i] for i in labels], alpha=0.5)

    cms = np.zeros(shape = (12, 2) )
    nums = np.zeros(12)
    for xy, label in zip(pca,labels):
        cms[label,:] += xy
        nums[label] += 1
    cms = np.array([cm/num for cm, num in zip(cms,nums)])

    for i in range(12):
        #scatter = plt.scatter(cms[i,0], cms[i,1], c = class_labels[i], marker='^', s=100)
        scatter = plt.scatter(cms[i,0], cms[i,1], c ='black', marker='*', s=100)
        _ = plt.text(cms[i,0], cms[i,1], s=class_names[i])
    
    if xlim:
        ax.set_xlim(xmin=xlim[0], xmax=xlim[1])
    if ylim:
        ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    
    if title:
        ax.set_title(title)

    ax.set_xlabel("{} PCA Component".format(x_comp+1))
    ax.set_ylabel("{} PCA Component".format(y_comp+1))
    ax.legend(loc='upper center', bbox_to_anchor = (0.5,1.2), ncol=6, fancybox=True, shadow = True, prop={'size':12})

    #plt.savefig(fname = outpath+str(epoch))
    #plt.show()
    return fig, pca

def run_KNN(embbedings, labels, tst_emb, tst_labels, neighbors=12):
    knn = KNN(n_neighbors=neighbors).fit(embbedings, labels)
    labs = knn.predict(tst_emb)
    return confusion_matrix(tst_labels, labs), labs

def nonn_samples(loader):
    # Prepare samples without any network
    embbedings = []
    labels_ = []
    names = []
    for j, data in enumerate(loader):
        inputs, labels, name = data
        embbedings.append(inputs)
        labels_.append(int(labels))
        names.append(name)
    embbedings = np.array(embbedings)
    labels_ = np.array(labels_)
    return embbedings, labels_, names

if __name__ == "__main__":
    main()