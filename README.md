# Artifact Identification
`SNICIT` achieves 6 ∼ 444× speed-ups on SDGC benchmarks and 1.36 ∼ 1.95× speed-ups on medium-scale sparse DNN applications (`MNIST` and `CIFAR-10`) over the previous years’ SDGC champions. Our computational artifact compares the performance of `SNICIT` and the previous years’ SDGC champions on 12 SDGC benchmarks and 4 medium-scale sparse DNNs targeting `MNIST` and `CIFAR-10`. It also demonstrates the decomposition of the runtime and the impact of threshold layer and batch size on performance. After compilation, our computational artifact generates two executable files, corresponding to experiments on SDGC benchmarks (Section 4.1) and beyond SDGC (Section 4.2) in the article, respectively. The artifact contains the dataset (input, medium-scale sparse DNNs) for experiments beyond SDGC benchmarks, and a script to download SDGC dataset (input, benchmark DNNs) from the official SDGC website. All the third-party dependencies are packed in the artifact. Our computational artifact can reproduce all the experiments mentioned in the paper. We will make it open-source to benefit both HPC and the machine learning community for accelerating sparse DNN inference.
 
 # Reproducibility of Experiments
 ## Time Consumption
 ## Storage Usage
 ## File Hierachy
 ## Environment
 ## Experimental Workflow
 
