# ILSVRC2012 ensembling experiments
 This project contains experiments performed with WeightedLDAEnsemble[^1] on ImageNet ILSVRC2012 dataset.
 Four neural networks were trained using the modified training script[^2] and then different subsets of these networks were combined.
 Ensembles training script is contained in the file ensemble_networks.py. Evaluation scripts are in the file process_results.py.
 
 ## Usage
 ### Ensemble training
 Ensemble training script is executed by
 ```
 $ python ensemble_networks.py -network_outputs path_to_networks_outputs -outputs_folder path_to_outputs_folder -min_ensemble_size 2 -device cuda
 ```
 The script expects two subfolders in the network_outputs folder named val_outputs and test_outputs, 
 containing networks outputs for samples on which the ensemble should be trained and tested, respectively.
 Both folders should also contain targets.npy file, containing correct class labels for samples in the folder.
 
 ### Results evaluation
 Results evaluation script is executed by 
 ```
 $ python process_results.py -network_outputs path_to_networks_test_outputs -ensemble_outputs path_to_ensemble_outputs -evaluation_output path_to_output_folder -device cuda
 ```
 The script produces three summary evaluations nets.csv, combins.csv and combins_ss.csv.
 nets.csv contains information about neural networks performance. combins.csv contains information about ensembles performance.
 conbins_ss.csv contain information about ensemble performance on different subsets of testing data. These subsets are formed according to which networks were correct on which sample.
 All combinations of networks correct/incorrect outputs are contained in these subsets. For example, first subset will be formed of the samples on which all the networks were correct,
 second subset will be formed of the samples on which all, but net1 were correct, and so on.
 
 [^1]: https://github.com/ReneFabricius/weighted_ensembles
 [^2]: https://github.com/ReneFabricius/ILSVRC2012_train
