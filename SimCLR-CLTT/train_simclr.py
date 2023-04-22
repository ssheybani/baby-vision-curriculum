from argparse import ArgumentParser

import pytorch_lightning as pl
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform, SimCLRTrainDataTransform)
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.loggers import TensorBoardLogger

from datamodules import ImageFolderDataModule, ImagePairsDataModule
from models.simclr import SimCLR
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def create_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Max number of epochs to train."
    )
    parser.add_argument(
        "--val_split",
        default=0.01,
        type=float,
        help="Percent (float) of samples to use for the validation split."
    )
    
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporally ordered image pairs."
    )
    parser.add_argument(
        "--window_size",
        #default=2,
        type=int,
        help="Size of sliding window for sampling temporally ordered image pairs."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--dataset_size",
        default=0,
        type=int,
        help="Subset of dataset"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="wandb dashboard project name"
    )
    parser.add_argument(
        "--seed_val",
        type=int,
        default=0,
        help="SEED VALUE"
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=['resnet34','resnet18','resnet_3blocks','resnet_2blocks','resnet_1block'],
        help="select architecture"
    )
    parser.add_argument(
        "--temporal_mode",
        type=str,
        choices=['2+images', '2images'],
        help="select how many images to push together"
    )
    
    return parser

def cli_main():

    parser = create_argparser()

    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()
    args.gpus = 1
    args.lars_wrapper = True
    args.arch = args.architecture
    #print("images are shuffled - ", args.shuffle)

    if args.temporal:
        dm = ImagePairsDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False, # shuffle is decided using arg flag
            drop_last=False, # changed from True to False becz of empty dataloader error
            val_split=args.val_split,
            window_size=args.window_size,
            temporal_mode=args.temporal_mode,
        )
    else:
        dm = ImageFolderDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False, # changed from True to False becz of empty dataloader error
            val_split=args.val_split,
            dataset_size=args.dataset_size,
        )

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=dm.size()[-1],
        # gaussian_blur=args.gaussian_blur,
        # jitter_strength=args.jitter_strength,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=dm.size()[-1],
        # gaussian_blur=args.gaussian_blur,
        # jitter_strength=args.jitter_strength,
    )

    # The SimCLR data transforms are designed to be used with datamodules
    # which return a single image. But ImagePairsDataModule returns
    # a pair of images.
    if isinstance(dm, ImagePairsDataModule):
        dm.train_transforms = dm.train_transforms.train_transform
        dm.val_transforms = dm.val_transforms.train_transform

    pl.seed_everything(args.seed_val)

    args.num_samples = dm.num_samples

    model = SimCLR(**args.__dict__)

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger("LOGS/simclr", name=f"{args.exp_name}")
   
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        logger=logger,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
    )

    #print(model)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
