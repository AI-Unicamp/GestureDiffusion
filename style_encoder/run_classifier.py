from style_encoder.parser_util import generate_args
from style_encoder.classifier import prep_data_and_model, run_samples, run_PCA
import os
import torch



def main():
    args = generate_args()
    network, valloader, device = prep_data_and_model(args, val_only=True)
    network = load_model(args, network, device)
    fig = run(network, valloader, device)
    fig.savefig(os.path.join(args.classifier_path, 'PCA.png'))

def load_model(args, network, device):
    # load parameters from file
    pth_name = os.path.basename(args.classifier_path) + '.pth'
    state_dict = torch.load(os.path.join(args.classifier_path, pth_name), map_location=device)
    network.load_state_dict(state_dict)
    return network


def run(network, valloader, device):
    embeddings, labels, samples = run_samples(network, valloader, device)
    fig = run_PCA(embeddings, labels, n_components=2, samples_names=samples)
    return fig

if __name__ == "__main__":
    main()