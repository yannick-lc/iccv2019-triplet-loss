import pandas as pd
import numpy as np

class ZslDataset:
    
    PATHS = {
        "awa2": '/scratch_ssd/yannick/Animals_with_Attributes2/preprocessed/',
        "cub": '/scratch_ssd/yannick/CUB_200_2011/preprocessed/',
        "apy": '/scratch_ssd/yannick/aPascalYahoo/preprocessed/',
        "sun": '/scratch_ssd/yannick/SUN/preprocessed/'
    }
    
    def __init__(self, df_classes, df, attributes, features):
        """
        N: number of samples in dataset
        C: number of classes
        D: dimension of visual features space
        K: dimension of semantic features space
        
        Parameters:
            df_classes: C-row pandas dataframe representing classes
                should contain columns 'class', 'class_name', and splits
            df: N-row pandas dataframe representing samples
                should contain columns 'class' and splits
            attributes: C x K numpy array containing semantic representation of class prototypes
            features: N x D numpy array containing visual features representation of samples
        """
        self.df_classes = df_classes
        self.df = df
        self.attributes = attributes
        self.features = features