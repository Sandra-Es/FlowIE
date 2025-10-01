from argparse import ArgumentParser

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

from utils.common import instantiate_from_config, load_state_dict


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
    to_unfreeze = ["layer4", "fc"]   # example for a ResNet
    for name, module in model.named_modules():
        if any(name.startswith(t) for t in to_unfreeze):
            for p in module.parameters():
                p.requires_grad = True

    num_params = sum([p.numel() for p in model.parameters()])
    num_train_params = sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
    print(f"\n\nTrainable Params: {num_train_params} / {num_params}\n\n")
    ###

    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)
    trainer.fit(model, datamodule=data_module)
    #trainer.test()


if __name__ == "__main__":
    main()