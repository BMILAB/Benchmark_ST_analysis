# Benchmark ST analysis


## Overview
This research paper focuses on benchmarking
computational methods for detecting spatial domains and domain-specific spatially variable genes (SVGs) 
from spatial transcriptomics data. The study comprehensively evaluates 19 computational methods, 
including 14 newly emerging deep learning methods, using 30 real-world datasets covering six spatial transcriptomics
technologies and 27 synthetic datasets. Rather than treating clustering and SVG analysis—two key downstream analyses 
of spatial transcriptomics—as separate issues, this paper considers the impact of clustering on SVG
recognition and establishes an integrated analysis workflow. Initially, the study assesses the performance 
of these methods in identifying spatial domains, focusing on accuracy, stability, generality, and scalability. 
Furthermore, the study conducts SVG identification on both real and simulated data, evaluating
the impact of spatial domains identified by different methods on domain-specific SVG detection. 
This approach provides a more holistic understanding of how clustering influences the identification of SVGs, 
offering valuable insights for the analysis of spatial transcriptomics data.



## Requirements
You'll need to install the following packages in order to run the codes.
* python==3.8
* torch>=1.8.0
* cudnn>=10.2
* numpy==1.22.3
* scanpy==1.9.1
* anndata==0.8.0
* rpy2==3.4.1
* pandas==1.4.2
* scipy==1.8.1
* scikit-learn==1.1.1
* torch-geometric==2.3.1
* scanpy==1.9.1
* scikit-misc==0.2.0
* scipy==1.10.1
* SpaGCN==1.2.7
* squidpy==1.1.2
<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Benchmarking methods
Here is a list of ST analysis tools with links to their respective documentation or GitHub repositories:

* [Louvain](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.louvain.html) (Wolf et al., 2018) and [Leiden](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html) (Traag et al., 2019) are community detection algorithms used for cell clustering in single-cell RNA sequencing analysis, with Leiden being an enhancement of Louvain.
* [Seurat](https://github.com/satijalab/seurat) (Satija et al., 2015) is an R package for single-cell RNA sequencing data analysis, offering tools for preprocessing, clustering, differential expression analysis, and visualization.

* [BayesSpace](https://github.com/edward130603/BayesSpace) (Zhao et al., 2021) employs a Bayesian framework to enhance data resolution and perform clustering by integrating spatial neighborhood information.
* [MUSE](https://github.com/AltschulerWu-Lab/MUSE) (Bao et al., 2022) utilizes auto-encoders to integrate H&E and gene expression data, with a focus on self-supervised learning and triple loss function training.

* [SpaCell](https://github.com/BiomedicalMachineLearning/SpaCell) (Tan et al., 2020) classifies cell types and disease stages by merging H&E image features with gene expression data through neural networks.

* [ConST](https://github.com/ys-zong/conST) (Zong et al., 2022) is an interpretable framework that extracts morphological features and captures spatial relationships for contrast learning.
* [STAGATE](https://github.com/QIFEIDKN/STAGATE) (Dong and Zhang, 2022) combines spatial location with gene expression data using a graph attention autoencoder for various downstream tasks.

* [SEDR](https://github.com/JinmiaoCHEnLab/SEDR) (Fu et al., 2021) integrates gene expression and spatial information to enhance clustering and visualization through deep self-encoders.

* [SpaceFlow](https://github.com/hongleir/SpaceFlow) (Ren et al., 2022) constructs a Spatial Expression Graph and uses GCN to generate embeddings for visualization and clustering.

* [Spatial-MGCN](https://github.com/cs-wangbo/Spatial-MGCN) (Wang et al., 2023) is a spatial clustering algorithm based on multi-view graph convolutional networks.

* [SCGDL](https://github.com/narutoten520/SCGDL) (Wang et al., 2023) uses RGGCNN for gene expression and spatial location encoding to generate spatial domains.

* [CCST](https://github.com/xiaoyeye/CCST) (Li et al., 2022) encodes spatial and gene expression information using the DGI network and PCA for clustering.

* [GraphST](https://github.com/JinmiaoCH&EnLab/GraphST) (Long et al., 2023) uses graph neural networks and contrast learning for spatial clustering.

* [SpaGCN](https://github.com/jianhuupenn/SpaGCN) (Hu et al., 2021) constructs a graph from gene expression and spatial location for clustering analysis.

* [stLearn](https://github.com/BiomedicalMachineLearning/stLearn) (Pham et al., 2020) standardizes gene expression values using spatial morphological information.

* [DeepST](https://github.com/JiangBioLab/DeepST) (Xu et al., 2022) integrates H&E, gene expression, and spatial data using a Domain Adversarial Neural Network.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Download all datasets used in this benchmark

The raw datasets used in this paper can be downloaded from the following websites. Specifically,
* **10X_151507~151676** : [Human Dorsolateral Pre-frontal Cortex ](http://research.libd.org/globus/jhpce_HumanPilot10x/index.html) -  Contains average 3499 spots and 33538 genes.
* **10X_BC** : [Human Breast Cancer (Block A Section 1) ](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0)  - Contains 3798 spots/cells and 36601 genes.
* **10X_MB** : [Mouse brain dataset](https://squidpy.readthedocs.io/en/stable/api/squidpy.datasets.visium_hne_adata.html)  - Mouse brain dataset with 2688 spots/cells and 18078 genes.
* **ST_PDAC_A** : [Pancreatic ductal epithelial tissue](https://doi.org/10.1038/s41587-019-0392-8)  - Contains 428 spots/cells and 19736 genes.
* **ST_Hip_1** : [Hippocampal region of mouse brain - section 1](https://data.mendeley.com/datasets/6s959w2zyr/1)  - Contains 601 spots and 18509 genes.
* **ST_Hip_2** : [Hippocampal region of mouse brain - section 2](https://data.mendeley.com/datasets/6s959w2zyr/1)  - Contains 649 spots and 18357 genes.
* **Stereo_AB** : [Axolotl Brain ](https://db.cngb.org/stomics/artista/)  - Contains 8243 spots/cells and 22144 genes.
* **Stereo_DPI** : [Axolotl Brain ](https://db.cngb.org/stomics/artista/)  - Dataset during the regeneration phase of the telencephalon two days post-injury (DPI) with 7668 spots/cells and 27324 genes.
* **Stereo_MB** : [Mouse Brain (Nervous Tissue) ](https://db.cngb.org/stomics/datasets/STDS0000234)  - Contains 2343 spots/cells and 24302 genes.
* **Slide-seqV2_MKD** : [Mouse Kidney with Diabetes](https://cellxgene.cziscience.com/collections/d74b6979-efba-47cd-990a-9d80ccf29055)  - Contains 18542 spots/cells and 20190 genes.
* **Slide-seqV2_MKN** : [Mouse Kidney](https://cellxgene.cziscience.com/collections/d74b6979-efba-47cd-990a-9d80ccf29055)  - Contains 16027 spots/cells and 19283 genes.
* **Slide-seqV2_embryo** : [Mouse Embryo ](https://cellxgene.cziscience.com/collections/d74b6979-efba-47cd-990a-9d80ccf29055)  - Contains 8425 spots/cells and 27554 genes.
* **STARmap_MVC** : [Mouse Brain Visual Cortex ](https://stagate.readthedocs.io/en/latest/index.html#)  - Contains 1207 spots/cells and 1020 genes.
* **STARmap_MPC** : [Mouse Brain Prefrontal Cortex](https://github.com/libedeutch/BOOST-HMI/blob/main/data)  -Contains 1549 spots/cells and 1020 genes.
* **STARmap_BZ5** : [Mouse Brain Medial Prefrontal Cortex - STARmap Resources](https://www.starmapresources.org/data)  - Contains 1049 spots/cells and 166 genes.
* **STARmap_AD** : [Mouse Brain with Alzheimer's disease](https://zenodo.org/records/7332091)  - Contains 7127 spots/cells and 2766 genes.
* **SeqFISH_embro_1** : [Mouse Embryo](https://github.com/MarioniLab/SpatialMouseAtlas2020)  - Contains 11026 spots/cells and 351 genes.
* **SeqFISH_embro_2** : [Mouse Embryo ](https://github.com/MarioniLab/SpatialMouseAtlas2020)  - Contains 19416 spots/cells and 351 genes.
* **SeqFish_plus_cortex** : [Mouse Brain Somatosensory Cortex](https://github.com/CaiGroup/seqFISH-PLUS)  - Contains 913 spots/cells and 10000 genes.

Due to github upload memory limitations, pre-processed real data as well as simulated data for this study can be downloaded from
Figshare (<https://figshare.com/projects/Benchmark_ST_analysis/234116>)

<p align="right">(<a href="#readme-top">back to top</a>)</p>






### 4. Tutorial
The main work is as follows:

  We take all the data and methods and integrate them into a unified framework. Firstly, you need to load the necessary dependency packages to ensure that each method is executed successfully, and then output the prediction results to the `Output` file. 
Following that, you should integrate the predicted labels and place them in the `SVG_indentified/Pred_label`  directory for the purpose of SVG identification. 

If you want to analyze your own data, please put it in the corresponding directory.
In addition, we provide a tutorial in the `Tutorial`  directory that demonstrates how to run the method.

For more details, please see our paper, thank you very much.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
