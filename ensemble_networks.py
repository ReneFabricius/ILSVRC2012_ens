import os
from itertools import combinations
import argparse

from weighted_ensembles.general_test import ensemble_general_test
from weighted_ensembles.SimplePWCombine import m1, m2, bc, m2_iter


def ensemble_networks():
    """
    Creates ensembles of all the possible subsets of networks found in network_outputs/val_outputs. Minimal size of
    the subset is given by min_ensemble_size. Ensemble outputs combining network outputs found in
    network_outputs/test_outputs are computed for each tested topl (given in code) and stored into outputs_folder.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-network_outputs', type=str, required=True, help='path to networks outputs folder')
    parser.add_argument('-outputs_folder', type=str, required=True, help='path to folder for storing outputs')
    parser.add_argument('-min_ensemble_size', default=2, type=int, help='minimal number of networks in ensemble')
    parser.add_argument('-test_normality', dest='test_normality', action='store_true',
                        help='enables normality testing for LDA predictors')
    parser.add_argument('-no_test_normality', dest='test_normality', action='store_false',
                        help='disables normality testing for LDA predictors')
    parser.add_argument('-double_precision', dest='double_precision', action='store_true',
                        help='enables double precision')
    parser.add_argument('-single_precision', dest='double_precision', action='store_false',
                        help='enables single precision')
    parser.add_argument('-load_models', dest='load_models', action='store_true',
                        help='enables model loading if available')
    parser.add_argument('-no_load_models', dest='load_models', action='store_false',
                        help='disables model loading')
    parser.set_defaults(test_normality=True, double_precision=False, load_models=False)
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    parser.add_argument('-topls', type=int, nargs="+", default=[5], help="list of topl values to test")
    args = parser.parse_args()

    train = "val_outputs"
    test = "test_outputs"

    output = "output"
    model = "model"
    targets = "targets.npy"
    order = "order.txt"
    topls = args.topls

    net_val_outputs = os.path.join(args.network_outputs, train)
    net_test_outputs = os.path.join(args.network_outputs, test)
    order_file_val = os.path.join(net_val_outputs, order)
    order_file_test = os.path.join(net_test_outputs, order)

    net_outputs = []
    for file in os.listdir(net_val_outputs):
        if file.endswith(".npy") and file != targets:
            net_outputs.append(file)
            print("Network output found: " + file)

    num_nets = len(net_outputs)
    for sss in range(args.min_ensemble_size, num_nets + 1):
        print("Testing " + str(sss) + " network ensembles")
        for sub_set in combinations(net_outputs, sss):
            for topl in topls:
                print("Testing topl: " + str(topl))
                sub_set_name = '_'.join([s[0:4] for s in sub_set]) + "_topl_" + str(topl)
                outputs_fold = os.path.join(args.outputs_folder, output + "_" + sub_set_name)
                models_fold = os.path.join(args.outputs_folder, model + "_" + sub_set_name)
                if not os.path.exists(outputs_fold):
                    os.makedirs(outputs_fold)
                if not os.path.exists(models_fold):
                    os.makedirs(models_fold)

                order_fl = open(order_file_val, 'w')
                order_fl.write('\n'.join(sub_set))
                order_fl.close()

                order_fl_test = open(order_file_test, 'w')
                order_fl_test.write('\n'.join(sub_set))
                order_fl_test.close()

                try:
                    if args.load_models and os.path.isfile(os.path.join(models_fold, 'models')):
                        ensemble_general_test(net_val_outputs, net_test_outputs, targets, order, outputs_fold,
                                              models_fold, [m1, m2, m2_iter, bc], combining_topl=topl, save_coefs=True,
                                              verbose=False, test_normality=args.test_normality, save_pvals=True,
                                              fit_on_penultimate=True, double_precision=args.double_precision,
                                              models_load_file=os.path.join(models_fold, 'models'))
                    else:
                        ensemble_general_test(net_val_outputs, net_test_outputs, targets, order, outputs_fold,
                                              models_fold, [m1, m2, m2_iter, bc], combining_topl=topl, save_coefs=True,
                                              verbose=False, test_normality=args.test_normality, save_pvals=True,
                                              fit_on_penultimate=True, double_precision=args.double_precision)

                finally:
                    os.remove(order_file_val)
                    os.remove(order_file_test)


if __name__ == "__main__":
    ensemble_networks()
