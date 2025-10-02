The datasets.py file need to be modified to be able to load the different datasets:
The function load(folder) in data.py will search in folder anyfile that named filename.dge.suff or filename.matrix.suff where suff is in {'.txt', '.csv', '.tsv', '.mtx', '.rds', '.h5', '.h5ad', '.h5ad', '.h5seurat'}
It will also look for any file following the regex '^(filename)([A-Za-z0-9\-\_\.]+)'. Among those files, those that finish with .barcodes.suff or .genes.suff or .obs.suff with suff in {'.txt', '.csv', '.tsv'} are also loaded and merged in the anndata where they should. If there are no gene names and the ENSEMBL ids are used, then the function tries to replace them using the genome annotations in annotations. Only Mouse and Human are supported.

For example, a structure as follows works for the Zheng2017 data

Zheng17/
  68k_pbmc.barcodes.tsv  
  68k_pbmc.matrix.mtx
  68k_pbmc.genes.tsv     
  68k_pbmc_celltypes.obs.txt
  
Ultimately, all the path to each of these folders should be included in the DATASETS dictionnary in the datasets.py file to be able to run all experiments
The raw data files can be accessed at the following urls:

Luecken2021 (multiome): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122
Zheng2017 (10xchromiumv1): https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis
Hravtin2016 (indrops): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE102827
Jensen2022 (smart-seq3xpress): https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11452
Macosko2015 (drop-seq): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63472
Stuart2019 (cite-seq): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3681518
Wang2020 (10xchromiumv3): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE111976

Note that the Luecken2017 needs to be separated between RNA and ATAC before using it
