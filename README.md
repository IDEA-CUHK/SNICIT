# Artifact Identification
`SNICIT` achieves 6 ∼ 444× speed-ups on SDGC benchmarks and 1.36 ∼ 1.95× speed-ups on medium-scale sparse DNN applications (`MNIST` and `CIFAR-10`) over the previous years’ SDGC champions. Our computational artifact compares the performance of `SNICIT` and the previous years’ SDGC champions on 12 SDGC benchmarks and 4 medium-scale sparse DNNs targeting `MNIST` and `CIFAR-10`. It also demonstrates the decomposition of the runtime and the impact of threshold layer and batch size on performance. After compilation, our computational artifact generates two executable files, corresponding to experiments on SDGC benchmarks (Section 4.1) and beyond SDGC (Section 4.2) in the article, respectively. The artifact contains the dataset (input, medium-scale sparse DNNs) for experiments beyond SDGC benchmarks, and a script to download SDGC dataset (input, benchmark DNNs) from the official SDGC website. All the third-party dependencies are packed in the artifact. Our computational artifact can reproduce all the experiments mentioned in the paper. We will make it open-source to benefit both HPC and the machine learning community for accelerating sparse DNN inference.
 
 # Reproducibility of Experiments
 ## Time Consumption
It takes about 6 hours including human effort to run all the experiments. Most of the time is spent on experiments on SDGC benchmarks (Section 4.1): downloading SDGC dataset (≈ 2 hours, depending on network speed), SDGC benchmark parameter preprocessing (≈ 2 hours, depending on computing power), and dataset file I/O (adding up to approximately 1 hour for all the experiments). This is why we **strongly** recommend running experiments beyond SDGC (Section 4.2) at first. Experiments beyond SDGC benchmarks consume about 1 hour including human effort. We have already packed the input dataset (`MNIST` and `CIFAR-10`) and the medium-scale sparse DNN parameters inside the artifact. And due to their relatively small size, file I/O will not take much time.

 ## Storage Usage
The artifact itself is not large (<150 MB). However, to run all the experiments, a total storage space of 185 GB is required. The SDGC dataset alone occupies 137.2 GB, and the temporary files generated from preprocessing SDGC dataset occupies 46.8 GB. Again, we strongly recommend running experiments beyond SDGC (Section 4.2) if you do not have adequate storage space to run experiments on SDGC benchmarks. The artifact (<150 MB) has everything you need to run experiments beyond SDGC.

 ## File Hierachy
 (a) `3rd-party`: It contains all the third-party dependencies (`CLI`, `Eigen`, and `Taskflow`).
 (b) `bin`: It contains the scripts for compiling the executables, automatically running the executables under different arguments, and automatically downloading SDGC dataset from the Internet. The compiled executables will also be placed in this folder
 (c) `dataset`:  It contains the dataset for the experiments.
 (d) `log`:  It contains output logs from the experiments.
 (e) `main`: It contains the two main function entrances for experiments on and beyond SDGC benchmarks.
 (f) `plot`: It contains a Python script for plotting heatmaps in **Figure 12**. The heatmaps generated from the script will be stored in this `plot/figs/`.
 (g) `scheduled_bm`: It contains the temporary files generated from preprocessing SDGC benchmark parameters.
 (h) `src`: It contains the source code for the experiments.
 ## Environment
 ## Experimental Workflow
 
