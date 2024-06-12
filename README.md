# PSCFA
Identification of multi-functional therapeutic peptides based on prototypical supervised contrastive learning
#  Introduction
In this paper, we propose PSCFA, a prototypical supervised contrastive learning with feature augmentation method for prediction of MFTP.The contributions of this work can be outlined as follows:  
(1)We propose a novel two-stage network structure for identification of MFTP, demonstrating that improved feature representation can facilitate better classifier learning. The network incorporates of a total loss function, integrated both contrastive loss and dice loss for feature learning, complemented by a dice loss for classifier learning.   
(2)We explore an effective prototypical supervised contrastive learning strategies to refine feature learning to boost classification performance for MFTP.  
(3)The feature augmentation strategy for tail labels can effectively improve classification performance of the model on classes with limited instances.  
(4)Our experimental results conclusively show that PSCFA consistently outperforms current leading methods when evaluated on MFTP datasets, underscoring its efficacy and superiority in the domain of computational peptide analysis.  

The framework of the PSCFA method for MFTP prediction is described as follows:  
![The framework of the PSCFA model](images/The%20framework%20of%20the%20PSCFA%20model.png "The framework of the PSCFA model")
#  Related Files  
#   PSCFA  
| FILE NAME       | DESCRIPTION                                                            |
|-----------------|------------------------------------------------------------------------|
| `pep_main.py`       | the main file of ETFC predictor  |
| `train.py`      | train model                                                            |
| `models.py`      | model construction                                                     |
| `evaluation.py` | evaluation metrics (for evaluating prediction results)                 |
| `dataset`       | data                                                                   |
| `result`        | results preserved during training.                          |
| `saved_models`        | modelspreserved during training.                          |
