library(Seurat)
library(stringr)
library(BayesSpace)
library(writexl)
library(SingleCellExperiment)
library(scater)

root_dir <- getwd()
sample_id <- list.files(path = root_dir,full.names = T)
for (i in sample_id) {
  time_seu_s <- Sys.time()
  print(i)
  print('Seurat processing...')
  count <- read.csv(file = file.path(i,'expression_data.csv'),header = T,row.names = 1)
  spatial <- read.csv(file = file.path(i,'spatial.csv'),row.names = 1)
  coord<- spatial
  colnames(coord) <- c('row','col')
  count <- t(count)
  label <- read.csv(file.path(i,"pred_label.csv"), row.names = 1,header = T)
  # filter cell/spots
  is_Zero <- which(colSums(count)==0)
  if (length(is_Zero)!=0) {
    count <- count[,-c(is_Zero)]
    coord <- coord[-c(is_Zero),]
  }

  sce <- SingleCellExperiment(assays = list(counts = count),colData = coord)
  if(!is.integer(count[1,1])){
    logcounts(sce) <- SingleCellExperiment::counts(sce)
  }else{
    sce <- scater::logNormCounts(sce)
  }


  dir.output <- file.path(i,'Seurat')

  if(!dir.exists(file.path(dir.output))){
    dir.create(file.path(dir.output), recursive = TRUE)
  }



  sp_data <- CreateSeuratObject(counts = count)

  # sctransform
  # sp_data <- SCTransform(sp_data,verbose = FALSE,variable.features.n = nrow(sp_data))
  sp_data <- SCTransform(sp_data,verbose = FALSE,variable.features.n = 3000)
  ### Dimensionality reduction, clustering, and visualization
  sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE, npcs = 50)

  sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:30)

  n_clusters=length(unique(label$Seurat))
  for(resolution in 10:200){
    sp_data <- FindClusters(sp_data, verbose = F, resolution = resolution/100)
    if(length(levels(sp_data@meta.data$seurat_clusters)) == n_clusters){
      break
    }
  }
  sp_data <- RunUMAP(sp_data, reduction = "pca", dims = 1:50)
  time_seu_e <- Sys.time()
  saveRDS(sp_data, file.path(dir.output, 'Seurat_final.rds'))


  dir.output <- file.path(i,'BayesSpace')
  if(!dir.exists(file.path(dir.output))){
    dir.create(file.path(dir.output), recursive = TRUE)
  }


  print('BayesSpace processing...')
  time_bay_s <- Sys.time()

  sp_bayes <- sce
  sp_bayes <- scater::logNormCounts(sp_bayes)

  set.seed(101)
  dec <- scran::modelGeneVar(sp_bayes)
  # top <- scran::getTopHVGs(dec, n = nrow(sp_bayes))
  top <- scran::getTopHVGs(dec, n = 3000)
  set.seed(102)
  sp_bayes <- scater::runPCA(sp_bayes, subset_row=top)

  ## Add BayesSpace metadata
  sp_bayes <- spatialPreprocess(sp_bayes, platform = "ST", skip.PCA=TRUE)


  ##### Clustering with BayesSpace
  q <- n_clusters  # Number of clusters
  d <- 50  # Number of PCs

  ## Run BayesSpace clustering
  set.seed(104)
  sp_bayes <- spatialCluster(sp_bayes, q=q, d=d, platform='ST',
                             nrep=50000, gamma=3, save.chain=TRUE)
  time_bay_e <- Sys.time()
  saveRDS(sp_bayes,file = file.path(dir.output,'model.rds'))

  print(paste('Seurat:',time_seu_e - time_seu_s,'BayesSpace:',time_bay_e - time_bay_s,sep = ' '))
}


root_dir <- getwd()
sample_id <- list.files(path = root_dir,full.names = T)
for (i in sample_id) {
  print(paste(i,'...'))
  sampleSub = i
  outpath = sampleSub
  seu <- readRDS(file.path(outpath,'Seurat','Seurat_final.rds'))
  label_seu <- data.frame(seu@meta.data[["seurat_clusters"]],row.names = colnames(seu))
  colnames(label_seu) <- 'predict'
  PCA_dim_seu <- seu@reductions[["pca"]]@cell.embeddings
  bay <- readRDS(file.path(outpath,'BayesSpace','model.rds'))
  PCA_dim_bay <- bay@int_colData@listData[["reducedDims"]]
  PCA_dim_bay <- data.frame(PCA_dim_bay@listData[["PCA"]])
  label_bay <- data.frame(bay@colData@listData[["spatial.cluster"]],row.names = colnames(bay))
  colnames(label_bay) <- 'predict'
  UMAP_dim_seu <- uwot::umap(PCA_dim_seu)
  UMAP_dim_bay <- uwot::umap(PCA_dim_bay)

  PCA_dim_seu <- cbind(row.names(PCA_dim_seu), PCA_dim_seu)
  UMAP_dim_seu <- cbind(row.names(UMAP_dim_seu), UMAP_dim_seu)
  label_seu <- cbind(row.names(label_seu), label_seu)

  write_xlsx(list("PCA" = data.frame(PCA_dim_seu),
                  "UMAP" = data.frame(UMAP_dim_seu),
                  "predict" = data.frame(label_seu)),
             path = file.path(outpath,"Seurat","label_embedd.xlsx"))

  PCA_dim_bay <- cbind(row.names(PCA_dim_bay), PCA_dim_bay)
  UMAP_dim_bay <- cbind(row.names(UMAP_dim_bay), UMAP_dim_bay)
  label_bay <- cbind(row.names(label_bay), label_bay)

  write_xlsx(list("PCA" = data.frame(PCA_dim_bay),
                  "UMAP" = data.frame(UMAP_dim_bay),
                  "predict" = data.frame(label_bay)),
             path = file.path(outpath,"BayesSPace","label_embedd.xlsx"))
}
