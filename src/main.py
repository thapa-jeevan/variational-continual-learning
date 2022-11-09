import argparse

from src.experiments import discriminative, permuted_mnist
from src.experiments import generative

# experiments from the VCL paper that can be carried out
EXP_OPTIONS = {
    'disc_p_mnist': permuted_mnist.permuted_mnist,
    'disc_s_mnist': discriminative.split_mnist,
    'disc_s_n_mnist': discriminative.split_not_mnist,
    'gen_mnist': generative.generate_mnist,
    'gen_not_mnist': generative.generate_not_mnist,
    'gen_mnist_classifier': generative.train_mnist_classifier,
    'gen_n_mnist_classifier': generative.train_not_mnist_classifier
}


def main(experiment='all'):
    # run all experiments
    if experiment == 'all':
        for exp in list(EXP_OPTIONS.keys()):
            print("Running", exp)
            EXP_OPTIONS[exp]()
    # select specific experiment to run
    else:
        EXP_OPTIONS[experiment]()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('experiment', help='Experiment to be run, can be one of: ' + str(list(EXP_OPTIONS.keys())))
    args = parser.parse_args()
    main(args.experiment)
