
import matplotlib.pyplot as plt
import warnings
from sklearn import set_config
import rpy2.rinterface_lib.callbacks
import logging
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
import anndata2ri
import os
from anndata import ImplicitModificationWarning

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
pandas2ri.activate()
anndata2ri.activate()

if 'R_LIBS_USER' not in os.environ.keys(): 
    os.environ['R_LIBS_USER'] = '/mnt/home/vchardes/.local/lib/R'


if 'R_LIBS_SITE' in os.environ.keys():
    robjects.globalenv['lib_site'] = os.environ['R_LIBS_SITE']
    robjects.r('.libPaths(lib_site)')
    
if 'R_LIBS_USER' in os.environ.keys(): 
    robjects.globalenv['lib_user'] = os.environ['R_LIBS_USER']
    robjects.r('.libPaths(lib_user)')
    

# TODO Install necessary R dependencies to run the R code at the moment the code is called
#library(scran)
#library(RColorBrewer)
#library(slingshot)
#library(monocle)
#library(gam)
#library(clusterExperiment)
#library(ggplot2)
#library(plyr)
#library(MAST)
ret = robjects.r('''local({r <- getOption("repos")
                  r["CRAN"] <- "https://cran.r-project.org" 
                  options(repos=r)
               })
           ''')



set_config(print_changed_only=False)
warnings.filterwarnings('ignore', category = ImplicitModificationWarning)
warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)


#Parameter for a figure with one axis only: figures should be assembled in inkscape
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['figure.dpi'] = 300
#plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['figure.figsize'] = (7.18/3,7.18/3)
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['lines.markersize'] = 3.0
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.01

plt.rcParams.update({# Use mathtext, not LaTeX
                     'text.usetex': False,
                     # Use the Computer modern font
                     'font.family': 'sans-serif',
                     'font.serif': 'cmr10',
                     'font.sans-serif': 'Nimbus Sans',
                     'axes.formatter.use_mathtext': True,
                     'mathtext.fontset': 'cm',
                     # Use ASCII minus
                     'axes.unicode_minus': False,
                     })

