import torchvision
import cv2
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

class CFG:
    version_note = 'v1_timm'

    root_folder = './'
    run_folds = [0] #[0,1,2,3,4]
    device = 'cuda:0'
    comet_api_key = 'zR96oNVqYeTUXArmgZBc7J9Jp' # change to your key
    comet_project_name = 'KaggleBreastCancer'
    im_size = 256

    num_workers=2
    backbone="tf_efficientnetv2_s"
    pretrained_weights = True
    gradient_checkpointing=False
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    
    resume = False
    resume_key = None
    epochs=30
    init_lr=3e-4
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=64
    weight_decay=0.01
    warmup_factor = 10
    fp16 = True
    save_best_only=True
    checkpoint_monitor = 'validate_pfbeta'

    ema = True
    ema_decay = 0.99

    clip_grad_norm = 10
    accumulation_steps = 1

    seed=67    
    sample = None
    patience = 10

CFG.metadata_file = f'{CFG.root_folder}/data/train_5folds.csv'
CFG.train_im_dir = f'{CFG.root_folder}/data/train/'
CFG.model_dir = f'{CFG.root_folder}/models'
CFG.valid_pred_folder = f'{CFG.root_folder}/valid_predictions'
CFG.submission_folder = f'{CFG.root_folder}/submissions'

# data augmentation and transformations
CFG.train_transforms = A.Compose(
        [   
            A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),

            A.Resize(height=CFG.im_size, width=CFG.im_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
        ],
        p=1.0,
    )


CFG.val_transforms = A.Compose(
        [
            A.Resize(height=CFG.im_size, width=CFG.im_size, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
        ],
        p=1.0,
       
    )