import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # class label at leaf

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth=1, min_sample_split: int = 2):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y)
        self.progress.close()

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        num_samples, nums_features = X.shape
        num_classes = len(np.unique(y))
        if (
            depth >= self.max_depth or
            num_samples < self.min_sample_split or
            num_classes == 1
        ):
            leaf_value = self._majority_class(y)
            return TreeNode(value=leaf_value)
    
        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            leaf_value = self._majority_class(y)
            return TreeNode(value=leaf_value)
        
        left_x, left_y, right_x, right_y = self._split_data(X, y, feat_idx, threshold)
        
        self.progress.update(1)
        left_child = self._build_tree(left_x, left_y, depth + 1)
        right_child = self._build_tree(right_x, right_y, depth + 1)
        return TreeNode(feature_index=feat_idx, threshold=threshold, left=left_child, right=right_child)
        # raise NotImplementedError

    def predict(self, X: np.ndarray)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        preds = [self._predict_tree(x, self.tree) for x in X]
        return np.array(preds)
        # raise NotImplementedError

    def _predict_tree(self, x, node: 'TreeNode'):
        # (TODO) Recursive function to traverse the decision tree
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)
        # raise NotImplementedError

    @staticmethod
    def _split_data(X: np.ndarray, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node 
        mask = X[:, feature_index] <= threshold
        left_X, right_X = X[mask], X[~mask]
        left_y, right_y = y[mask], y[~mask]
        # raise NotImplementedError
        return left_X, left_y, right_X, right_y

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        best_gain = -1
        best_feat, best_thresh = None, None
        current_entropy = self._entropy(y)
        n_features = X.shape[1]
        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                left_x, left_y, right_x, right_y = self._split_data(X, y, feat_idx, threshold)
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gain = self._information_gain(y, left_y, right_y, current_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_thresh = feat_idx, threshold
        # raise NotImplementedError
        return best_feat, best_thresh

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        # (TODO) Return the entropy
        probs = np.bincount(y) / len(y)
        probs = probs[probs > 0]
        # raise NotImplementedError
        return -np.sum(probs * np.log2(probs))
    
    def _information_gain(self, parent, left_child, right_child, parent_entropy):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        gain = parent_entropy - (weight_left * self._entropy(left_child) + weight_right * self._entropy(right_child))
        return gain
    
    @staticmethod
    def _majority_class(y):
        return np.bincount(y).argmax()

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="extract", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            features.extend(feats.cpu().numpy())
            labels.extend(lbls.numpy())
    features = np.vstack(features)
    labels = np.array(labels)
    # raise NotImplementedError
    return features, labels

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    model.eval()
    features = []
    paths = []
    with torch.no_grad():
        for imgs, names in tqdm(dataloader, desc="extract", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            features.extend(feats.cpu().numpy())
            paths.extend(names)
    features = np.vstack(features)
    # raise NotImplementedError
    return features, paths