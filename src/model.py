import numpy as np
from torch import nn
from torchvision import models
import timm
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel
from .metric import pfbeta_torch

class BCModel(BaseModel):
    def __init__(self, backbone_name, backbone_pretrained=None, n_classes=1, device='cpu'):
        super(BaseModel, self).__init__()
        
        self.backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained)
        
        if 'nfnet' in backbone_name:
            clf_in_feature = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(clf_in_feature, n_classes)
        elif 'resnet' in backbone_name:
            clf_in_feature = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(clf_in_feature, n_classes)
        else:
            clf_in_feature = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(clf_in_feature, n_classes)

        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

        self.to(self.device)

    def forward(self, X):
        y = self.backbone(X)
        return y

    def step(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)
        y_pred = self(X).view(-1)
        loss = self.criterion(y_pred, y)
        return loss, y_pred

    def training_step(self, train_batch):
        X, y = train_batch
        loss, logits = self.step(X, y)
        y_prob = logits.sigmoid()
        return {'loss': loss, 'preds':y_prob, 'labels':y}

    def validation_step(self, val_batch):
        X, y = val_batch
        loss, logits = self.step(X, y)
        y_prob = logits.sigmoid()
        return {'loss': loss, 'preds':y_prob, 'labels':y}

    def compute_metrics(self, outputs):
        all_preds = np.concatenate([out['preds'].detach().cpu().numpy() for out in outputs])
        all_labels = np.concatenate([out['labels'].detach().cpu().numpy() for out in outputs])
        auc = float(roc_auc_score(y_true=all_labels, y_score=all_preds))
        pfbeta = pfbeta_torch(all_labels, all_preds)
        return auc, pfbeta

    def training_epoch_end(self, training_step_outputs):
        train_auc, pfbeta = self.compute_metrics(training_step_outputs)
        return {'AUC': train_auc, 'pfbeta': pfbeta}
        
    def validation_epoch_end(self, validation_step_outputs):
        val_auc, pfbeta = self.compute_metrics(validation_step_outputs)
        return {'AUC': val_auc, 'pfbeta': pfbeta}


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)