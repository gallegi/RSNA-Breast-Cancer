import os
import argparse
import importlib
import pandas as pd
import torch

from warmup_scheduler import GradualWarmupScheduler
from torch.optim import AdamW

from src.general import seed_everything
from src.dataset import BCDataset
from src.model import BCModel
from src.trainer import Trainer

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='config',
                    help='config file to run an experiment')
parser.add_argument('--im_dir', type=str, default='data/train',
                    help='path to training image folder')
parser.add_argument('--model_dir', type=str, default='./models',
                    help='config file to run an experiment')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
CFG = config_module.CFG

CFG.model_dir = args.model_dir
CFG.train_im_dir = args.im_dir

CFG.output_dir_name = CFG.version_note + '_' + CFG.backbone.replace('/', '_') 
CFG.output_dir = os.path.join(args.model_dir, CFG.output_dir_name)

if not torch.cuda.is_available() and (not torch.backends.mps.is_available()) :
    CFG.device = 'cpu'

print(CFG.device)

df = pd.read_csv(CFG.metadata_file)

os.makedirs(CFG.output_dir, exist_ok=True)


import linecache
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


tracemalloc.start()



for val_fold in CFG.run_folds:
    print(f'================= Training fold {val_fold} ================')
    seed_everything(CFG.seed) # set seed each time a fold is run

    train_df = df[df['fold']!=val_fold].reset_index(drop=True)
    val_df = df[df['fold']==val_fold].reset_index(drop=True)

    if(CFG.sample):
        train_df = train_df.sample(CFG.sample).reset_index(drop=True)
        val_df = val_df.sample(CFG.sample).reset_index(drop=True)

    CFG.num_train_examples = len(train_df)

    # Defining DataSet
    train_dataset = BCDataset(train_df, CFG.train_im_dir, CFG.train_transforms)
    val_dataset = BCDataset(val_df, CFG.train_im_dir, CFG.val_transforms)

    batch_size = CFG.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,
                                               pin_memory=True,drop_last=True,num_workers=CFG.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,num_workers=CFG.num_workers,
                                               shuffle=False,pin_memory=True,drop_last=False)

    # Model
    model = BCModel(CFG.backbone, CFG.pretrained_weights, device=CFG.device)
    if CFG.torch_compile:
        model = torch.compile(model, mode="reduce-overhead")

    # Optimizer and scheduler
    optim = AdamW(model.parameters(), betas=CFG.betas, lr=CFG.init_lr/CFG.warmup_factor, weight_decay=CFG.weight_decay)

    num_training_steps = CFG.epochs * len(train_loader)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, CFG.epochs-1)
    lr_scheduler = GradualWarmupScheduler(optim, multiplier=CFG.warmup_factor, total_epoch=1, after_scheduler=scheduler_cosine)
    
    trainer = Trainer(CFG, model, train_loader, val_loader,
                    optimizer=optim, lr_scheduler=lr_scheduler, fold=val_fold)

    trainer.fit()

snapshot = tracemalloc.take_snapshot()
display_top(snapshot)
