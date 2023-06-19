# SNICIT: Accelerating Sparse Neural Network Inference via Compression at Inference Time on GPU
Sparse deep neural network (DNN) has become an important technique for reducing the inference cost of large DNNs. However, computing large sparse DNNs is very challenging because inference iterations can incur highly irregular patterns and unbalanced loads. To address this challenge, the recent HPEC Graph Challenge seeks novel high-performance inference methods for large sparse DNNs. Despite the rapid progress over the past four years, solutions have largely focused on static model compression or sparse multiplication kernels, while ignoring dynamic data compression at inference time which can achieve significant yet untapped performance benefits. Consequently, we propose `SNICIT`, a new GPU algorithm to accelerate large sparse DNN inference via compression at inference time. `SNICIT` leverages data clustering to transform intermediate results into a sparser representation that
largely reduces computation over inference iterations. Evaluated on both HPEC Graph Challenge benchmarks and conventional DNNs (`MNIST`, `CIFAR-10`), SNICIT achieves 6 ∼ 444× and 1.36 ∼ 1.95× speed-ups over the previous champions, respectively.

# Artifact Identification
`SNICIT` achieves 6 ∼ 444× speed-ups on SDGC benchmarks and 1.36 ∼ 1.95× speed-ups on medium-scale sparse DNN applications (`MNIST` and `CIFAR-10`) over the previous years’ SDGC champions. Our computational artifact compares the performance of `SNICIT` and the previous years’ SDGC champions on 12 SDGC benchmarks and 4 medium-scale sparse DNNs targeting `MNIST` and `CIFAR-10`. It also demonstrates the decomposition of the runtime and the impact of threshold layer and batch size on performance. After compilation, our computational artifact generates two executable files, corresponding to experiments on SDGC benchmarks (Section 4.1) and beyond SDGC (Section 4.2) in the article, respectively. The artifact contains the dataset (input, medium-scale sparse DNNs) for experiments beyond SDGC benchmarks, and a script to download SDGC dataset (input, benchmark DNNs) from the official SDGC website. All the third-party dependencies are packed in the artifact. Our computational artifact can reproduce all the experiments mentioned in the paper. We will make it open-source to benefit both HPC and the machine learning community for accelerating sparse DNN inference.
 
 # Reproducibility of Experiments
 ## Time Consumption
It takes about 5 hours including human effort to run all the experiments. Most of the time is spent on experiments on SDGC benchmarks (Section 4.1): downloading SDGC dataset (≈ 2 hours, depending on network speed), SDGC benchmark parameter preprocessing (≈ 2 hours, depending on computing power), and dataset file I/O (adding up to approximately 1 hour for all the experiments). This is why we **strongly** recommend running experiments beyond SDGC (Section 4.2) at first. Experiments beyond SDGC benchmarks only consume several minutes including human effort. We have already packed the input dataset (`MNIST` and `CIFAR-10`) and the medium-scale sparse DNN parameters inside the artifact. And due to their relatively small size, file I/O will not take much time.

 ## Storage Usage
The artifact itself is not large (<150 MB). However, to run all the experiments, a total storage space of 185 GB is required. The SDGC dataset alone occupies 137.2 GB, and the temporary files generated from preprocessing SDGC dataset occupies 46.8 GB. Again, we strongly recommend running experiments beyond SDGC (Section 4.2) if you do not have adequate storage space to run experiments on SDGC benchmarks. The artifact (<150 MB) has everything you need to run experiments beyond SDGC.

 ## File Hierachy
 (a) `3rd-party`: It contains all the third-party dependencies (`CLI`, `Eigen`, and `Taskflow`).
 
 (b) `bin`: It contains the scripts for compiling the executables, automatically running the executables under different arguments, and automatically downloading SDGC dataset from the Internet. The compiled executables will also be placed in this folder.
 
 (c) `dataset`:  It contains the dataset for the experiments.
 
 (d) `log`:  It contains output logs from the experiments.
 
 (e) `main`: It contains the two main function entrances for experiments on and beyond SDGC benchmarks.
 
 (f) `plot`: It contains a Python script for plotting heatmaps in **Figure 12**. The heatmaps generated from the script will be stored in this `plot/figs/`.
 
 (g) `scheduled_bm`: It contains the temporary files generated from preprocessing SDGC benchmark parameters.
 
 (h) `src`: It contains the source code for the experiments.
 
 ## Environment
 (a) A CentOS 8 x86 64-bit machine, with 8 Intel i7-11700 CPU cores at 2.5 GHz, one RTX A6000 48 GB GPU, and 128 GB RAM.
 
 (b) `g++` compiler version 8.5.0, with `-std=c++17`.
 
 (c) `nvcc` version 12.0, with `-std=c++17`.
 
 (d) `Python` version 3.9.13, `NumPy` version 1.23.4, `seaborn` version 0.12.2, and `Matplotlib` version 3.6.1 (Optional: for generating heatmaps in **Figure 12**).

 ## Experimental Workflow
 (a) Please clone the repository.
 
 (b) Next, we jump into `SNICIT-main` folder. We suggest running experiments beyond SDGC benchmarks first. please go to directory `bin/` first and run the script `compile.sh`.
 
 `∼/$ cd SNICIT-main`
 
`∼/SNICIT-main$ cd bin`

`∼/SNICIT-main/bin$ ./compile.sh`

If you encounter "`permission denied`" error, please use this command.

`∼/SNICIT-main/bin$ chmod u+x *.sh`

After compilation, two executable files named beyond and SDGC will be generated under `bin/`. We begin with experiments beyond SDGC (i.e. executable file beyond) first. If you run SDGC at this stage, there will be file I/O errors because SDGC dataset has not yet been downloaded.

(c) After obtaining beyond, you can run it by specifying the arguments. You can specify mode by `-m`. The available modes are `BF` (`BF-2019` in the article), `SNIG` (`SNIG-2020` in the article), and `SNICIT` (default). You can specify the threshold layer by `-t`. Any integer in 0 ∼ l − 1 can be threshold. Batch size is specified by `-b`, with choices of 1000, 2000, 2500, 5000, or 10000, and sparse DNN ID is specified by `-k`, with choices of `A`, `B`, `C`, or `D`. For example, if you want to run on DNN A using `SNICIT` with a threshold of 8 and a batch size of 5000, please use the following command.

`∼/SNICIT-main/bin$ ./beyond -m SNICIT -t 8 -b 5000 -k A`

You can run this command on DNNs A and D by using `SNICIT` to obtain the runtime decomposition data for Figure 10 from terminal output message (runtime of the four stages shown in **Figure 2**).

(d) To obtain the data for **Table 4** and **Figure 11**, you can run the script `beyond_tab4_fig11.sh`. 

`∼/SNICIT-main/bin$ ./beyond_tab4_fig11.sh`

This script can automatically run the three methods on the four medium-scale sparse DNNs (A, B, C, and D in **Table 4**). The output log file is `log/beyond/tab4_fig11.txt`. The log contains inference accuracy, runtime, and average post-convergence latency for each run.

(e) To obtain the heatmaps in **Figure 12**, please use the following command to run `beyond_heatmap.sh`. 

`∼/SNICIT-main/bin$ ./beyond_heatmap.sh`

This script can automatically conduct a grid search for all t in \[0, l) with a step size of 2 and all B in {1000, 2000, 2500, 5000, 10000} on DNNs A, B, C, and D by using `SNIG-2020` and `SNICIT`. Output logs can be found in `log/beyond/`. Then, please go to directory `plot/` and find a Python file plot_beyond.py. It can parse the output logs and extract `SNICIT`'s speed-up over `SNIG-2020` and accuracy loss to plot the heatmaps. The heatmaps can be found in folder `plot/figs/` (`Python`  environment and packages required).

`∼/SNICIT-main/bin$ cd ../plot`

`∼/SNICIT-main/plot$ python plot_beyond.py`

(f) Now, we move on to the experiment on SDGC benchmarks. We switch back to folder `bin/` again, and run script `get_SDGC_dataset.sh` with `--all` to download the SDGC dataset.

`∼/SNICIT-main/plot$ cd ../bin`

`∼/SNICIT-main/bin$ ./get_SDGC_dataset.sh --all`

(g) We run executable SDGC under `bin/`, which was generated from Step (b). You can run it by specifying the mode (`-m` `BF`, `SNIG`, `XY`, or `SNICIT`), network width (`-n` 1024, 4096, 16384, or 65536), network depth (`-l` 120, 480, or 1920), threshold (`-t` an integer in 0 ∼ l − 1), and batch size (`-b` an integer that is a factor of 60000). For example, if you want to run `SNICIT` on benchmark 1024-1920 with a threshold of 30 and a batch size of 60000, please use the following command.

`∼/SNICIT-main/bin$ ./SDGC -m SNICIT -n 1024 -l 1920 -t 30 -b 60000`

You can also run the command on DNNs 1024-120, 4096-120, 16384-120, and 65536-120 by using `SNICIT` to obtain the data for **Figure 7** from terminal output message (runtime of the four stages shown in **Figure 2**). When you run SDGC using `XY-2021` or `SNICIT` on a certain benchmark for the first time, it will take a long time to conduct network parameter preprocessing (adding up to approximately 2 hours for all the benchmarks). To save time, we save the preprocessed parameters in `scheduled_bm/`, so that we can avoid redundant preprocessing on the same benchmarks every time we run the program. Please also notice that when you run SDGC using `XY-2021` or `SNICIT` on benchmarks with 65536 neurons per layer, you may encounter a memory overflow error if your GPU has a small memory capacity. Please decrease the batch size to fit into GPU memory.

(h) To obtain the data for **Table 3** and **Figure 6**, please run the script `SDGC_tab3_fig6.sh`. This script can automatically run the four methods on SDGC benchmarks. The output log file is `log/SDGC/tab3_fig5.txt`. It contains the runtime for every run and average post-convergence latency for `XY-2021` and `SNICIT` runs. However, we did not include `XY-2021` and `SNICIT` for benchmarks with 65536 neurons per layer, because we do not know the batch size that can fit your GPU memory. Please run SDGC with `XY-2021` and `SNICIT` on benchmarks with 65536 neurons manually with an appropriate batch size for your GPU.

(i) To obtain data for **Figure 8** and **Figure 9**, please run the script `SDGC_fig89.sh`

`∼/SNICIT-main/bin$ ./SDGC_fig89.sh`

This script can automatically run SNICIT on DNNs 1024-120, 4096-120, and 16384-120 under different thresholds. It can also automatically run `XY-2021` and `SNICIT` on 1024-120, 4096-120, and 16384-120 under different batch sizes. The output log file is `log/SDGC/fig89.txt`, which contains the runtime for each run. However, we did not include `XY-2021` and `SNICIT` on DNN 65536-1920 (**Figure 9 (d)**), because we do not know the batch size that can fit your GPU memory. Please run `SDGC` with `XY-2021` and `SNICIT` on DNN 65536-1920 manually with appropriate batch sizes for your GPU to obtain the runtime data for **Figure 9 (d)**.

# Reference
+ [A GPU Implementation of the Sparse Deep Neural Network Graph Challenge](https://doi.org/10.1109/HPEC.2019.8916223)
+ [A Novel Inference Algorithm for Large Sparse Neural Network using Task Graph Parallelism](https://ieeexplore.ieee.org/abstract/document/9286218)
+ [SNIG-2020 and BF-2019 implementation](https://github.com/dian-lun-lin/SNIG)
+ [Fast Sparse Deep Neural Network Inference with Flexible SpMM Optimization Space Exploration](https://ieeexplore.ieee.org/document/9622791)
+ [XY-2021 implementation](https://github.com/CGCL-codes/Graphchallenge21)


