"""
Implementation of method in [arxiv link]

Notations:

N: number of samples (may be different between train and test)  
C: number of classes  
D: dimension of visual space  
K: dimension of semantic space  
H: dimension of learned representation (projected X and S)  

X: usually N x D matrix of visual samples  
S: usually C x K matrix of semantic prototypes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sklearn
import numpy as np
import pandas as pd


# FLEXIBLE SEMANTIC MARGIN (Section 3.2)

def pairwise_mahalanobis(S1, S2, Cov_1=None):
    """
        S1: C1 x K matrix (torch.FloatTensor)
          -> C1 K-dimensional semantic prototypes
        S2: C2 x K matrix (torch.FloatTensor)
          -> C2 K-dimensional semantic prototypes
        Sigma_1: K x K matrix (torch.FloatTensor)
          -> inverse of the covariance matrix Sigma; used to compute Mahalanobis distances
          by default Sigma is the identity matrix (and so distances are euclidean distances)
        
        returns an C1 x C2 matrix corresponding to the Mahalanobis distance between each element of S1 and S2
        (Equation 5)
    """
    if S1.dim() != 2 or S2.dim() != 2 or S1.shape[1] != S2.shape[1]:
        raise RuntimeError("Bad input dimension")
    C1, K = S1.shape
    C2, K = S2.shape
    if Cov_1 is None:
        Cov_1 = torch.eye(K)
    if Cov_1.shape != (K, K):
        raise RuntimeError("Bad input dimension")
    
    S1S2t = S1.matmul(Cov_1).matmul(S2.t())
    S1S1 = S1.matmul(Cov_1).mul(S1).sum(dim=1, keepdim=True).expand(-1, C2)
    S2S2 = S2.matmul(Cov_1).mul(S2).sum(dim=1, keepdim=True).t().expand(C1, -1)
    return torch.sqrt(torch.abs(S1S1 + S2S2 - 2. * S1S2t) + 1e-32)  # to avoid numerical instabilities

def distance_matrix(S, mahalanobis=True, mean=1., std=0.5):
    """
        S: C x K matrix (numpy array)
          -> K-dimensional prototypes of C classes
        mahalanobis: indicates whether to use Mahalanobis distance (uses euclidean distance if False)
        mean & std: target mean and standard deviation
        
        returns a C x C matrix corresponding to the Mahalanobis distance between each pair of elements of S
        rescaled to have approximately target mean and standard deviation while keeping values positive
        (Equation 6)
    """
    Cov_1 = None
    if mahalanobis:
        Cov, _ = sklearn.covariance.ledoit_wolf(S) # robust estimation of covariance matrix
        Cov_1 = torch.FloatTensor(np.linalg.inv(Cov))
    S = torch.FloatTensor(S)
    
    distances = pairwise_mahalanobis(S, S, Cov_1)
    
    # Rescaling to have approximately target mean and standard deviation while keeping values positive
    max_zero_distance = distances.diag().max()
    positive_distances = np.array([x for x in distances.view(-1) if x > max_zero_distance])
    emp_std = float(positive_distances.std())
    emp_mean = float(positive_distances.mean())
    distances = F.relu(std * (distances - emp_mean) / emp_std + mean)
    emp_std = float(distances.std())
    emp_mean = float(distances.mean())
    distances = F.relu(std * (distances - emp_mean) / emp_std + mean)
    return distances


# PARTIAL NORMALIZATION (Section 3.3)

def partial_normalization(X, gamma):
    """
        X: N x H matrix (torch.FloatTensor)
          -> projected visual (or semantic) samples
        gamma: scalar between 0 and 1
          -> normalization coefficient
        
        returns N x H matrix corresponding to X matrix where each row has been partially normalize
        (Equation 8)
    """
    partial_norms = 1. / (gamma * (X.norm(p=2, dim=1) - 1) + 1)
    partial_norms = partial_norms.view(-1, 1)
    X = partial_norms * X
    return X


# RELEVANCE WEIGHTING (Section 3.4)

def class_weights(X_c):
    """
        X_c: N x D matrix of N D-dimensional visual samples, assumed to belong to the same class c
        
        returns the corresponding relevance weights
    """
    mean_vector = X_c.mean(axis=0).reshape(-1, 1)
    distances_to_mean_vector = np.sqrt((X_c.T - mean_vector).T.dot(X_c.T - mean_vector).diagonal())
    distribution = stats.norm(*scipy.stats.norm.fit(distances_to_mean_vector))
    return 1. - distribution.cdf(distances_to_mean_vector)

def relevance_weigths(X, Y):
    """
        X: N x D matrix of N D-dimensional visual samples
        Y: N dimensional vector of classes
        
        returns an N-dimensional vector corresponding to relevance weights of each visual samples
    """
    weights = np.zeros(Y.shape[0])
    classes = sorted(set(Y))
    for c in classes:
        indexes_c = np.where(Y == c)
        X_c = X[indexes_c]
        weigths_c = class_weights(X_c)
        weights[indexes_c] = weigths_c
    return weights


# FINAL MODEL (Section 3.5)

def flexible_triplet_loss(X_theta, S_psi, Y, V, D_tilde):
    """
        X_theta: N x H matrix (torch.FloatTensor)
          -> projected visual features
        S_psi: C x H matrix (torch.FloatTensor)
          -> projected semantic features
        Y: N x C binary matrix (torch.LongTensor)
          -> labels
        V: N-dimensional vector (torch.FloatTensor)
          -> relevance weights
        Dtilde: C x C (torch.FloatTensor)
          -> semantic distance between each class
          
        returns the corresponding triplet loss
        (Equation 13, without regularization omega)
    """
    N, H = X_theta.size()
    C, _ = S_psi.size()
    if DEVICE == "cpu":
        Y = Y.type(torch.FloatTensor)
    else:
        Y = Y.type(torch.cuda.FloatTensor)
    
    pairwise_compatibilities = X_theta.mm(S_psi.t()) # all the f(x_n, s_c) (Equation 3)
    prototype_compatibilities = (Y * pairwise_compatibilities).sum(dim=1).view(-1, 1).expand(-1, C) # all the f(x_n, s_y) (Equation 3)
    margin = D_tilde.unsqueeze(0).expand(N, -1, -1) * Y.unsqueeze(2).expand(-1, -1, C) 
    margin = margin.sum(dim=1) # flexible semantic margin
    
    triplet_losses = F.relu(margin + pairwise_compatibilities - prototype_compatibilities) # (Equation 12)
    triplet_losses = (1. - Y) * triplet_losses # keeping only c != yn (in Equation 13)
    triplet_losses = V.view(-1, 1) * triplet_losses # weighting by relevance
    loss = triplet_losses.sum() / (N * C)
    return loss

class Projection(nn.Module):
    """
        Represents a linear projection from one space (visual or semantic) to another (semantic or common space)
        Projections are partially normalized (cf. Section 3.3)
    """
    
    def __init__(self, d_input, d_embedding, gamma=1.0):
        super(Projection, self).__init__()
        self.gamma = gamma
        self.fc1 = nn.Linear(d_input, d_embedding, bias=True)
        
    def norm(self):
        """
            returns average norm of parameters
        """
        norms = (
            self.fc1.weight.norm(p=1) + self.fc1.bias.norm(p=1)
        )
        size = (
            self.fc1.weight.shape[0] * self.fc1.weight.shape[1] + self.fc1.bias.shape[0]
        )
        return norms / size

    def forward(self, x):
        x = self.fc1(x)
        x = partial_normalization(x, self.gamma)
        return x

class FlexibleTripletLoss(AbstractModel):
    """
        Represents the final model
    """
    
    def __init__(self, params=None):
        # Default parameters
        self.params = {
            # Hyperparameters
            "lambda": 0., # regularization, Equation 13
            "mu_dtilde": 1.0, # mean of flexible margin, Equation 6
            "sigma_dtilde": 0.5, # standard deviation of flexible margin, Equation 6
            "gamma": 1.0, # partial normalization, Equation 8
            "setting": "thetapsi", # mapping of visual features (and semantic prototypes), Section 3.5
            
            # Other options (not hyperparameters)
            "epochs": 50,
            "learning_rate": 1e-3,
            "batch_size": 1000,
            "optimizer": optim.Adam,
            "num_workers": 4,
            "seed": 42L,
            "loss_multiplier": 1e3,
            "verbose": False,
            
            # Ablation study
            "mahalanobis_distance": True,
            "relevance_weighting": True
            # to disable partial normalization, set gamma to 0
            # to disable flexible semantic margin, set sigma_dtilde to 0 (and possible mu_dtilde to 1)
        }
        
        # Overriding default parameters if specified
        if params is not None:
            for param in params:
                self.params[param] = params[param]
            
    def fit(self, X, Y, S):
        """
            X: N x D matrix (numpy array)
              -> visual training features
            Y: N-dimensional vector (numpy array)
              -> labels
            S: C x K matrix (numpy array)
              -> training prototypes
        """        
        torch.manual_seed(self.params["seed"])
        np.random.seed(self.params["seed"])
        
        # Relevance weighting, Section 3.4
        if self.params["relevance_weighting"]: 
            V = torch.FloatTensor(relevance_weigths(X, Y))
        else:
            V = torch.ones(X.shape[0])
        
        N, D = X.shape
        C, K = S.shape
        H = K # embedding dimension is the dimension of the semantic space, Section 3.5
        
        X = torch.FloatTensor(X)
        Y_ = np.zeros([N, C])
        for n, c in enumerate(Y):
            Y_[n, c] = 1
        Y = torch.LongTensor(Y_)
        self.S = Variable(torch.FloatTensor(S).to(DEVICE))
        
        # Flexible semantic margin, Section 3.3
        self.D_tilde = Variable(distance_matrix(
            S,
            mahalanobis=self.params["mahalanobis_distance"],
            mean=self.params["mu_dtilde"], std=self.params["mu_dtilde"]*self.params["sigma_dtilde"]
        ).to(DEVICE))
        
        dataset = torch.utils.data.TensorDataset(X, Y, V)
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.params["batch_size"], shuffle=True, num_workers=self.params["num_workers"]
        )
        
        # Setting (theta or theta + psi, Section 3.5)
        self.visual_projection = Projection(D, H, gamma=self.params["gamma"]).to(DEVICE)
        self.semantic_projection = Projection(K, H, gamma=1.0).to(DEVICE) # we always normalize projection of S
        
        if self.params["setting"] == "thetapsi": # theta + psi
            self.optimizer = self.params["optimizer"](
                params=list(self.visual_projection.parameters()) + list(self.semantic_projection.parameters()),
                lr=self.params["learning_rate"], weight_decay=0.
            )
        else: # only theta
            self.optimizer = self.params["optimizer"](
                params=list(self.visual_projection.parameters()),
                lr=self.params["learning_rate"], weight_decay=0.
            )
        
        self.__train()
        
    def __train(self):
        """
            trains the model for the specified number of epochs with the specified hyperparameters (and options)
        """
        for epoch in range(self.params["epochs"]):
            if self.params["verbose"]:
                print "EPOCH %i" % epoch
            for i, (inputs, labels, weights) in enumerate(self.loader):
                
                N, D = inputs.shape
                _, C = labels.shape
                
                X = Variable(inputs.to(DEVICE))
                Y = Variable(labels.to(DEVICE))
                V = Variable(weights.to(DEVICE))

                self.optimizer.zero_grad()
                
                X_theta = self.visual_projection(X)
                regularization_loss = self.visual_projection.norm()
                
                if self.params["setting"] == "thetapsi":
                    S_psi = self.semantic_projection(self.S)
                    regularization_loss = regularization_loss + self.semantic_projection.norm()
                else:
                    S_psi = self.S
                
                loss = (
                    self.params["loss_multiplier"] * flexible_triplet_loss(X_theta, S_psi, Y, V, self.D_tilde)
                    + self.params["lambda"] * regularization_loss
                ) # Equation 13
                loss.backward()
                self.optimizer.step()
            
                if self.params["verbose"]:
                    print "Loss: %.2f" % loss.item()
    
    def predict(self, X, S):
        """
            X: N x D matrix (numpy array)
              -> visual test features
            S: C x K matrix (numpy array)
              -> test prototypes
              
            Note: N (number of features) and C (number of classes) are typically not the same as in fit
        """
        X = Variable(torch.FloatTensor(X).to(DEVICE))
        S = Variable(torch.FloatTensor(S).to(DEVICE))
        
        X_theta = self.visual_projection(X)
        if self.params["setting"] == "thetapsi":
            S_psi = self.semantic_projection(S)
        else:
            S_psi = S
        
        probabilities = X_theta.mm(S_psi.t())
        return probabilities.data.cpu().numpy()
