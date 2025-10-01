from argparse import ArgumentParser

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

from utils.common import instantiate_from_config, load_state_dict

""""
Game Plan

Main Idea : We can be sure that using a better coarse image generator (like a high res network)
            can significantly improve results, but this comes at a compute, and memory cost. So,
            our aim is to work with the current lightweight coarse image generator (SwinIR), along 
            with a minor classical compute input (like edge, face keypoints, etc) which might provide
            comparable results to the former. If so, the gains would be achieved at a fraction of the cost
            but with comparable performance to a high res network.

STAGE 1

- Ablation 1 : Train (Fusion before encoder)
- Ablation 2 : Train (Fusion before encoder) + (Injection modules)
- Ablation 3 : Train (Concat after encoder) + (Injection modules). 

Note: When fusing after encoder, we define and train a separate encoder 
just for edge latents which is then concatenated with coarse latent.

STAGE 2

Once these are done, out of the three, we choose the best finetuning method.
Then try finetuning with the face keypoint (instead of edge) as the extra input.

STAGE 3

Select a not so lightweight high res network, swap out SwinIR and produce results. No finetuning needed.
"""


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    pl.seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    # TODO: resume states saved in checkpoint.
    if config.model.get("resume"):
        weights = torch.load(config.model.resume, map_location="cpu")
        '''new_weights = {}
        for k in weights['state_dict']:
            
            
            if 'lora' not in k:
                new_weights[k] = weights['state_dict'][k]
        weights['state_dict'] = new_weights'''
        load_state_dict(model, weights, strict=False)
        
        
        #load_state_dict(model.preprocess_model, torch.load('/home/user001/zwl/data/flowir_work_dirs/swin_derain0/lightning_logs/version_6/checkpoints/step=69999.ckpt', map_location="cpu"), strict=True)

    #TODO ADDED : Freezing and training modules
    #Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze specific submodules by name
    to_unfreeze = ["edge_fusion"]   # example for a ResNet
    for name, module in model.named_modules():
        if any(name.startswith(t) for t in to_unfreeze):
            for p in module.parameters():
                p.requires_grad = True

    num_params = sum([p.numel() for p in model.parameters()])
    num_train_params = sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
    print(f"\n\nTrainable Params: {num_train_params:,} / {num_params:,}\n\n")
    ###

    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)
    trainer.fit(model, datamodule=data_module)
    #trainer.test()


if __name__ == "__main__":
    main()