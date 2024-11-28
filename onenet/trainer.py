import os
import time

import matplotlib.pyplot as plt

plt.switch_backend("agg")

import torch
import torch.nn as nn
import torch.optim as optim
from ruamel.yaml import YAML
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datasets.dataloaders import create_dataloader
from onenet.networks.onenet_network import OneNet


class Trainer:
    available_datasets = ("COCO", "VOC", "PET", "PET2", "BRAIN", "HEART", "SPLEEN", "LUNG")
    available_models = {"SEGONE": OneNet, "ONENET": OneNet}

    def __init__(self, opts, cuda):
        self.data_opts = opts["data"]
        self.train_opts = opts["train"]
        self.model_opts = opts["model"]

        self.device = torch.device(f"cuda:{cuda}" if self.train_opts["cuda"] and torch.cuda.is_available() else "cpu")

        # Load Data
        assert self.data_opts["name"] in self.available_datasets
        assert self.data_opts["resolution"][0] % 32 == 0 and self.data_opts["resolution"][1] % 32 == 0
        self.train_loader = create_dataloader(
            self.data_opts["datapath"],
            self.data_opts["name"],
            split=self.data_opts["train"],
            batch_size=self.train_opts["batch_size"],
            img_size=self.data_opts["resolution"],
            num_workers=self.train_opts["num_workers"],
        )
        self.val_loader = create_dataloader(
            self.data_opts["datapath"],
            self.data_opts["name"],
            split=self.data_opts["val"],
            batch_size=self.train_opts["batch_size"],
            img_size=self.data_opts["resolution"],
            num_workers=self.train_opts["num_workers"],
        )

        # Define model
        assert self.model_opts["name"] in self.available_models
        self.model = self.available_models[self.model_opts["name"]](self.model_opts)
        self.model.to(self.device)

        if not self.model_opts["name"] == "RESNET":
            print("Initializing Weight")
            self.initialize_weights()

        # If pretrain model exists, load
        if self.train_opts["load_weights"] is not None:
            self.load_model()
        elif self.train_opts["load_encoder"] is not None:
            self.load_encoder()

        # Set training params
        self.optim = optim.Adam(self.model.parameters(), self.train_opts["learning_rate"])
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optim, self.train_opts["lr_step"], 0.1)

        self.num_classes = self.model_opts["channel_out"]
        background_scaling = 8
        weights = [
            1 / self.num_classes
            + (background_scaling - 1) / (background_scaling * self.num_classes * (self.num_classes - 1))
        ] * (self.num_classes - 1)
        weights.insert(0, 1 / (background_scaling * self.num_classes))
        print(f"Loss weights: {weights}")
        self.weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        self.criteria = nn.CrossEntropyLoss(weight=self.weights_tensor)

        # Make dir and save options used
        self.save_dir = os.path.join(
            self.train_opts["save_dir"], f"{self.model_opts["name"]}_{self.data_opts["name"]}_{int(time.time())}"
        )
        os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, "config.yaml"), "wb") as f:
            yaml = YAML()
            yaml.dump(opts, f)

        # Loggers
        os.makedirs(os.path.join(self.save_dir, "images"))
        self.train_logger = SummaryWriter(os.path.join(self.save_dir, "train_log"))
        self.val_logger = SummaryWriter(os.path.join(self.save_dir, "val_log"))

    def process_batch(self, inputs):
        images, targets = inputs

        images = images.to(self.device)
        outputs = self.model(images)

        targets = targets.to(self.device)
        losses = self.criteria(outputs[-1], targets)

        return outputs, losses

    def train(self):
        self.model.train()

        best_val_loss = float("inf")
        best_model_weights = None
        counter = 0
        patience = 15
        early_stop_counter = 0
        early_stop_thres = 5
        total_loss = 0
        train_count = 0

        for self.epoch in range(self.train_opts["epoch"]):
            print(f"Epoch: {self.epoch}")

            for inputs in tqdm(self.train_loader):
                outputs, losses = self.process_batch(inputs)
                self.optim.zero_grad()
                losses.backward()
                self.optim.step()

                total_loss += losses
                train_count += 1

            losses = total_loss / train_count

            self.lr_scheduler.step()
            val_losses = self.val()

            self.train_logger.add_scalar("loss", losses, self.epoch)
            self.val_logger.add_scalar("loss", val_losses, self.epoch)
            print(f"Train loss: {losses}, Val loss: {val_losses}")

            if val_losses < best_val_loss:
                best_model_weights = self.model.state_dict()
                best_val_loss = val_losses
                counter = 0
                early_stop_counter = 0
                print(f"Best val loss at {val_losses}")
            else:
                counter += 1
                print(f"Val loss not improved. Patience: {patience - counter}")

            if counter >= patience:
                self.model.load_state_dict(best_model_weights)
                counter = 0
                early_stop_counter += 1
                print(f"Reverting Weight. Early Stop Counter: {early_stop_counter}/{early_stop_thres}")

            if early_stop_counter >= early_stop_thres:
                print(f"Early Stop with best val loss: {best_val_loss}")
                break

            if (self.epoch + 1) % self.train_opts["save_frequency"] == 0:
                self.save_model()

    def val(self):
        self.model.eval()

        if self.data_opts["name"] in ("BRAIN", "HEART", "SPLEEN", "LUNG"):
            if self.epoch == 0:
                self.val_iter = iter(self.val_loader)
            try:
                inputs = next(self.val_iter)
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                inputs = next(self.val_iter)

            with torch.no_grad():
                outputs, losses = self.process_batch(inputs)
                self.plot_mask(inputs[0], inputs[1], outputs[-1])
                del outputs

        else:
            total_loss = 0
            counter = 0
            with torch.no_grad():
                for inputs in tqdm(self.val_loader):
                    outputs, losses = self.process_batch(inputs)
                    total_loss += losses
                    counter += 1
                self.plot_mask(inputs[0], inputs[1], outputs[-1])
            losses = total_loss / counter
            self.model.train()

        return losses

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def save_model(self):
        weight_save_dir = os.path.join(self.save_dir, "model")
        weight_save_path = os.path.join(weight_save_dir, "weights_{}.pth".format(self.epoch))

        if not os.path.exists(weight_save_dir):
            os.makedirs(weight_save_dir)
        torch.save(self.model.state_dict(), weight_save_path)

        optim_save_path = os.path.join(weight_save_dir, "adam_{}.pth".format(self.epoch))
        torch.save(self.optim.state_dict(), optim_save_path)

    def load_model(self):
        # Load model weights
        print(f"Loading weights at {self.train_opts["load_weights"]}")
        try:
            self.model.load_state_dict(torch.load(self.train_opts["load_weights"], weights_only=True))
        except:
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.train_opts["load_weights"], weights_only=True)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        # Load optimizer
        if self.train_opts["load_optimizer"] is not None:
            print(f"Loading optimizer at {self.train_opts["load_optimizer"]}")
            self.optim.load_state_dict(torch.load(self.train_opts["load_optimizer"], weights_only=True))
        else:
            print("Warning: loading weights without optimizer")

    def load_encoder(self):
        print(f"Loading encoder weights at {self.train_opts["load_encoder"]}")
        try:
            self.model.encoder.load_state_dict(torch.load(self.train_opts["load_encoder"], weights_only=True))
        except:
            encoder_dict = self.model.encoder.state_dict()
            pretrained_dict = torch.load(self.train_opts["load_encoder"], weights_only=True)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
            encoder_dict.update(pretrained_dict)
            self.model.encoder.load_state_dict(encoder_dict)

    def plot_mask(self, images, targets, masks):
        """Given tensors, save as plotted images"""
        images = images.cpu().detach().numpy().transpose(0, 2, 3, 1)
        targets = targets.cpu().detach().numpy()
        masks = masks.cpu().detach()
        masks = torch.argmax(masks, dim=1)

        fig, axs = plt.subplots(2, 6)
        cmap = plt.get_cmap("viridis", self.model_opts["channel_out"])
        for i in range(2):
            for j in range(2):
                if len(images) <= i * 2 + j:
                    break
                axs[i, j * 3].imshow(images[i * 2 + j])
                axs[i, j * 3 + 1].imshow(targets[i * 2 + j], cmap=cmap, vmin=0, vmax=self.model_opts["channel_out"])
                axs[i, j * 3 + 2].imshow(masks[i * 2 + j], cmap=cmap, vmin=0, vmax=self.model_opts["channel_out"])
                axs[i, j * 3].axis("off")
                axs[i, j * 3 + 1].axis("off")
                axs[i, j * 3 + 2].axis("off")

                axs[i, j * 3].set_xticks([])
                axs[i, j * 3].set_yticks([])
                axs[i, j * 3 + 1].set_xticks([])
                axs[i, j * 3 + 1].set_yticks([])
                axs[i, j * 3 + 2].set_xticks([])
                axs[i, j * 3 + 2].set_yticks([])
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        fig.savefig(
            os.path.join(self.save_dir, "images", f"{self.epoch}.png"), dpi=1600, bbox_inches="tight", pad_inches=0
        )
        plt.close(fig)
