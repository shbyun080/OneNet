import argparse
from collections import defaultdict

import torch
from ruamel.yaml import YAML
from tqdm import tqdm

from datasets.dataloaders import create_dataloader
from onenet.networks.onenet_network import OneNet
from onenet.utils.eval import Evaluator

available_models = {"SEGONE": OneNet, "ONENET": OneNet}


def evaluate_network(cfg, cuda_num):
    # Load options
    with open(cfg) as cfg_file:
        yaml = YAML(typ="safe")
        opts = yaml.load(cfg_file)
    data_opts = opts["data"]
    train_opts = opts["train"]
    model_opts = opts["model"]

    # Set device
    device = torch.device(f"cuda:{cuda_num}" if train_opts["cuda"] and torch.cuda.is_available() else "cpu")

    # Load Data
    val_loader = create_dataloader(
        data_opts["datapath"],
        data_opts["name"],
        split=data_opts["val"],
        batch_size=train_opts["batch_size"],
        img_size=data_opts["resolution"],
        num_workers=train_opts["num_workers"],
    )

    assert model_opts["name"] in available_models
    model = available_models[model_opts["name"]](model_opts)
    model.to(device)

    print(f"Loading weights at {train_opts["load_weights"]}")
    try:
        model.load_state_dict(torch.load(train_opts["load_weights"], weights_only=True))
    except:
        print("Unpacking weights...")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(train_opts["load_weights"], weights_only=True)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.eval()

    def process_batch(inputs):
        images, targets = inputs

        images = images.to(device)
        outputs = model(images)

        targets = targets.to(device)
        evaluator = Evaluator(outputs[-1], targets, num_classes=model_opts["channel_out"])
        results = evaluator.evaluate_all()
        return results

    counter = 0
    total_results = defaultdict(lambda: 0)
    with torch.no_grad():
        for inputs in tqdm(val_loader):
            results = process_batch(inputs)
            for metric, value in results.items():
                if isinstance(value, torch.Tensor):
                    value = float(torch.mean(value))
                total_results[metric] += value
            counter += 1
    for metric in total_results:
        total_results[metric] /= counter

    return total_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    results = evaluate_network(args.cfg, args.cuda)
    for metric, value in results.items():
        print(f"{metric}: {value}")
    for metric, value in results.items():
        print(value, end=",")
    print()
