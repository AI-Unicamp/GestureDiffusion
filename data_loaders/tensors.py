import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})
    if 'mfcc' in notnone_batches[0]:
        mfccbatch = [b['mfcc'] for b in notnone_batches]
        mfccbatch = torch.cat(mfccbatch, dim=0)
        cond['y'].update({'mfcc': mfccbatch})
    if 'audio' in notnone_batches[0]:
        audiobatch = [b['audio'] for b in notnone_batches]
        audiobatch = torch.cat(audiobatch, dim=0)
        cond['y'].update({'audio': audiobatch})
    if 'seed' in notnone_batches[0]:
        seedbatch = [b['seed'].unsqueeze(0) for b in notnone_batches]
        seedbatch = torch.cat(seedbatch, dim=0)
        cond['y'].update({'seed': seedbatch})
    return motion, cond

# an adapter to our collate func
def gg_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[0].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[1], #b[0]['caption']
        'lengths': b[2],
        'audio': torch.tensor(b[3]).unsqueeze(0), # [seqlen] -> [1, seqlen]
        'mfcc': torch.tensor(b[4].T).float().unsqueeze(1).unsqueeze(0), # [seqlen, mfcc] -> [mfcc, 1, seqlen]
        'seed': torch.tensor(b[5].T).float().unsqueeze(1), # [n_seed_poses, J] -> [J, 1, n_seed_poses]
    } for b in batch]
    return collate(adapted_batch)

