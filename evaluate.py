import os
import argparse
import importlib
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from src.model import  BCModel
from src.dataset import BCDataset
from src.metric import pfbeta_torch

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config',
                    help='config file to run an experiment')
parser.add_argument('--fold', type=int, default=0,
                    help='fold to evaluate')
parser.add_argument('--im_dir', type=str, default='data/train',
                    help='path to training image folder')
parser.add_argument('--weight', type=str, default='models/v1_efficientnet_v2_s/fold0_best.pth',
                    help='trained weight file')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
CFG = config_module.CFG

CFG.train_im_dir = args.im_dir

CFG.output_dir_name = CFG.version_note + '_' + CFG.backbone.replace('/', '_') 
CFG.output_dir = os.path.join(CFG.model_dir, CFG.output_dir_name)

if not torch.cuda.is_available() and (not torch.backends.mps.is_available()) :
    CFG.device = 'cpu'

# Load metadata
df = pd.read_csv(CFG.metadata_file)

# Load model
model = BCModel(CFG.backbone, CFG.pretrained_weights, device=CFG.device)
print(model.load_state_dict(torch.load(args.weight, map_location=CFG.device)['model'], strict=True))
model.eval() # important

if CFG.sample is not None:
    df = df.sample(CFG.sample).reset_index(drop=True)

df['prediction_id'] = df.apply(lambda row: f"{row['patient_id']}_{row['laterality']}", axis=1)
train_df = df[df.fold != args.fold].reset_index(drop=True)
val_df = df[df.fold == args.fold].reset_index(drop=True)

val_dataset = BCDataset(val_df, CFG.train_im_dir, CFG.val_transforms)

batch_size = CFG.batch_size
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,num_workers=CFG.num_workers,
                                            shuffle=False,pin_memory=True,drop_last=False)

# Predict
val_preds = []
for batch in tqdm(val_loader, total=len(val_loader)):
    X, y = batch
    X = X.to(CFG.device)
    with torch.no_grad():
        y_prob = model(X).sigmoid().view(-1).cpu().numpy()
        val_preds.append(y_prob)

val_preds = np.concatenate(val_preds)
val_df.loc[:, 'prob'] = val_preds

print('FBeta image level:', pfbeta_torch(val_df.cancer, val_df.prob))
print('AUC image level:', roc_auc_score(val_df.cancer, val_df.prob))

val_df_grouped = val_df.groupby('prediction_id').max().reset_index()

print('FBeta prediction id level:', pfbeta_torch(val_df_grouped.cancer, val_df_grouped.prob))
print('AUC prediction id level:', roc_auc_score(val_df_grouped.cancer, val_df_grouped.prob))

os.makedirs(CFG.valid_pred_folder, exist_ok=True)
val_df.to_csv(os.path.join(CFG.valid_pred_folder, CFG.output_dir_name +  f'_valid_fold{args.fold}' + '.csv'), index=False)

