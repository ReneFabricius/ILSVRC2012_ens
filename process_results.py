import torch
import numpy as np
import os
from os import path, listdir
import pandas as pd
import re
import argparse

from weighted_ensembles.predictions_evaluation import compute_acc_topk, get_correctness_masks


def evaluate_results():
    """
    Computes top1 and top5 accuracies of neural networks found in network_outputs and of ensembles
    found in ensemble_outputs. These accuracies are stored into nets.csv and combins.csv respectively.
    Also computes accuracies of ensembles on subsets of testing data, where these subsets are formed according to
    correctness of constituting neural networks predictions. For example, first subsets will be formed of samples for
    which all of the constituting nns predicted correctly according to top1, second subsets will constitute of samples,
    where net1 was wrong, but the remaining ones were correct and so on.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-network_outputs', type=str, required=True, help='path to networks test outputs folder')
    parser.add_argument('-ensemble_outputs', type=str, required=True, help='path to ensembles test outputs folder')
    parser.add_argument('-evaluation_output', type=str, required=True, help='path to outputs folder')
    parser.add_argument('-device', type=str, default='cpu', help='device on which to execute the script')
    args = parser.parse_args()

    targets = "targets.npy"

    # Discovers present
    network_sets = set()
    for subfold in os.walk(args.ensemble_outputs):
        fold_name = path.split(subfold[0])[1]
        fold_name_split = fold_name.split('_')
        if fold_name_split[0] != "output":
            continue

        netw_set = frozenset(fold_name_split[1:-2])
        network_sets.add(netw_set)

    # Load targets and network predictions, compute accuracies
    tar = torch.from_numpy(np.load(path.join(args.network_outputs, targets)))
    num_images = tar.shape[0]
    computed_accuracies = [1, 5]
    net_predictions = {}
    nets_df = pd.DataFrame(columns=('net', *['top' + str(k) for k in computed_accuracies]))
    print("Processing nets folder {}".format(args.network_outputs))
    for f in listdir(args.network_outputs):
        if path.splitext(f)[1] == '.npy' and f != targets:
            print("Found network {}".format(f))
            cur_net = torch.from_numpy(np.load(path.join(args.network_outputs, f)))
            accuracies = [compute_acc_topk(tar, cur_net, k) for k in computed_accuracies]
            net_abrv = path.splitext(f)[0][:4]
            nets_df.loc[len(nets_df)] = [net_abrv, *accuracies]
            net_predictions[net_abrv] = cur_net

    nets_df.to_csv(path.join(args.evaluation_output, "nets.csv"), index=False)

    # Compute standard accuracies of ensembles
    methods = ['bc', 'm1', 'm2']
    comb_df = pd.DataFrame(columns=('method', 'topl', *net_predictions.keys(),
                                    *['top' + str(k) for k in computed_accuracies]))
    ptrn = r'output_(' + '|'.join([n_abr + "_" for n_abr in net_predictions.keys()]) + ')+topl_\d+'

    print("Processing combin folder {}".format(args.ensemble_outputs))
    for fold in listdir(args.ensemble_outputs):
        if path.isdir(path.join(args.ensemble_outputs, fold)) and re.search(ptrn, fold) is not None:
            print("Found combin output {}".format(fold))
            fold_split = fold.split('_')
            topl = int(fold_split[-1])
            cur_nets = fold_split[1:-2]
            for m in methods:
                pred = torch.from_numpy(np.load(path.join(args.ensemble_outputs, fold, "prob_" + m + ".npy")))
                accuracies = [compute_acc_topk(tar, pred, k) for k in computed_accuracies]
                comb_df.loc[len(comb_df)] = [m, topl, *[1 if net in cur_nets else 0 for net in net_predictions.keys()],
                                             *accuracies]

    comb_df.to_csv(path.join(args.evaluation_output, "combins.csv"), index=False)

    # Create top1 correctness masks for nets
    net_cor_masks = {}
    for net in net_predictions:
        cor_m = get_correctness_masks(net_predictions[net], tar, [1])
        net_cor_masks[net] = cor_m

    net_pred_keys = net_predictions.keys()
    del net_predictions
    # Create masks for net sets
    net_sets_masks = {}
    for st in network_sets:
        set_list = sorted(list(st))
        # Contains top1 correctness masks in rows for nets from set
        nets_cor = torch.cat([net_cor_masks[na].unsqueeze(0) for na in set_list], 0)
        masks = torch.zeros([2]*len(set_list) + [num_images], dtype=torch.bool)
        for cor_comb in range(2**len(set_list)):
            bin_comb = ('{0:0' + str(len(set_list)) + 'b}').format(cor_comb)
            mask_ind = [[int(b)] for b in bin_comb]
            mask_tens = torch.tensor(mask_ind)
            # Inverts correctness masks which should be false and computes logical and over the rows
            masks[mask_ind] = torch.prod(nets_cor == mask_tens, 0).type(torch.bool)

        net_sets_masks[st] = masks

    # Compute subset accuracies
    comb_ss_df = pd.DataFrame(columns=('method', 'topl', *net_pred_keys,
                                    *[na + "_cor" for na in net_pred_keys],
                                    *['top' + str(k) for k in computed_accuracies]))
    print("Processing combin folder {}".format(args.ensemble_outputs))
    for fold in listdir(args.ensemble_outputs):
        if path.isdir(path.join(args.ensemble_outputs, fold)) and re.search(ptrn, fold) is not None:
            print("Found combin output {}".format(fold))
            fold_split = fold.split('_')
            topl = int(fold_split[-1])
            cur_nets = sorted(fold_split[1:-2])
            cur_nets_set = frozenset(cur_nets)
            nets_cor = torch.cat([net_cor_masks[na].unsqueeze(0) for na in cur_nets], 0)
            for m in methods:
                pred = torch.from_numpy(np.load(path.join(args.ensemble_outputs, fold, "prob_" + m + ".npy")))
                ens_cor_masks = get_correctness_masks(pred, tar, computed_accuracies)
                for cor_comb in range(2 ** len(cur_nets)):
                    bin_comb = ('{0:0' + str(len(cur_nets)) + 'b}').format(cor_comb)
                    mask_ind = [[int(b)] for b in bin_comb]
                    mask = net_sets_masks[cur_nets_set][mask_ind].squeeze()
                    cur_ens_cor_masks = ens_cor_masks[:, mask]
                    cur_accur = torch.true_divide(torch.sum(cur_ens_cor_masks, 1), torch.sum(mask).item())

                    comb_ss_df.loc[len(comb_ss_df)] = [m, topl, *[1 if net in cur_nets else 0 for net in net_pred_keys],
                                                 *[-1 if net not in cur_nets else int(bin_comb[cur_nets.index(net)]) for net in net_pred_keys],
                                                *cur_accur.tolist()]

    comb_ss_df.to_csv(path.join(args.evaluation_output, "combins_ss.csv"), index=False)


if __name__ == "__main__":
    evaluate_results()