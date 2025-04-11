import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_confusion_matrix
from accelerate import Accelerator
from tqdm import tqdm
import wandb

from dataset import IMDbDataset
from model import GPT, NdGPT

class Trainer():
    def __init__(
            self, model, dataset,
            path_to_model,
            val_dataset=None,
            batch_size=32,
            num_epochs=10, save_every=1
    ):
        assert (num_epochs % save_every == 0)

        self.accelerator = Accelerator(
            split_batches=True,
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim = model.configure_optimizers()
        self.model, self.optim, self.dataloader = self.accelerator.prepare(model, optim, dataloader)

        self.val_dataloader = None
        if val_dataset is not None:
            self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        self.num_epochs = num_epochs
        self.save_every = save_every
        self.path = path_to_model
    @property
    def device(self):
        return self.accelerator.device
    
    def save(self, step=None):
        data = {
            'model': self.accelerator.get_state_dict(self.model),
            'optim': self.optim.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None
        }
        path = f"{self.path}.pt" if step is None else f"{self.path}_{step}.pt"
        torch.save(data, path)

    def load(self, step=None):
        path = f"{self.path}.pt" if step is None else f"{self.path}_{step}.pt"
        data = torch.load(path, map_location=self.device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.optim.load_state_dict(data['optim'])

        if self.accelerator.scaler is not None and data['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        block_times = []
        block_mem = []

        epoch = 0
        with tqdm(
            initial = epoch, total = self.num_epochs,
            disable = not self.accelerator.is_main_process
        ) as pbar:
            while epoch < self.num_epochs:
                for input_data, label in self.dataloader:
                    self.optim.zero_grad()
                    _, loss, time, mem = self.model(input_data, label)
                    self.accelerator.backward(loss)
                    self.optim.step()
                    if self.accelerator.is_main_process:
                        block_times.append(time*1000) # saved in ms
                        block_mem.append(mem/(1024*1024)) # saved in MB
                        wandb.log({"loss": loss, "time": time*1000, "memory": mem/(1024*1024)})
                        pbar.set_description(f'Loss: {loss:.8f}')
                        if (epoch+1) % self.save_every == 0:
                            self.save(epoch+1)
                epoch += 1
                pbar.update(1)
            wandb.log({"average_time": sum(block_times)/len(block_times), "average_memory": sum(block_mem)/len(block_mem)})

    @staticmethod
    def evaluate_batch(pred_logits, targ_labels):
        pred_labels = torch.argmax(pred_logits, dim=-1)
        num_correct = torch.sum(pred_labels == targ_labels).item()
        confusion = binary_confusion_matrix(pred_labels, targ_labels)
        return len(targ_labels), num_correct, confusion

    def evaluate(self):
        if self.val_dataloader is not None:
            n = 0
            accuracy = 0
            confusion_matrix = torch.zeros((2,2), device=self.device)
            for input_data, targ_labels in tqdm(self.val_dataloader):
                pred_logits, *_ = model(input_data)
                b, correct, confusion = self.evaluate_batch(pred_logits, targ_labels)
                n += b
                accuracy += correct
                confusion_matrix += confusion
            stats = {
                "Total Number" : n,
                "Accuracy" : accuracy/n,
                "TP" : confusion_matrix[1,1],
                "FP" : confusion_matrix[0,1],
                "FN" : confusion_matrix[1,0],
                "TN" : confusion_matrix[0,0],
            }

            for text, value in stats.items():
                print(f"{text}: {value}")
            return stats
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    ACTIVATE_LOGGING = True
    wandb_api = "ca7f0e161908db6645a6ec4a4b0a1714a3b2131c"
    wandb.login(key=wandb_api)
    wandb.init(project="NdLinear", mode="online" if ACTIVATE_LOGGING else "disabled")
    
    set_seed(42)

    full_dataset = IMDbDataset("IMDbData.csv")
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [0.8, 0.2],
        torch.Generator().manual_seed(42) # keeps train test split the same every run
    )

    model = NdGPT(full_dataset.vocab_size, full_dataset.max_length, 256, 2)
    trainer = Trainer(
        model, train_dataset, path_to_model="trained_weights/ndlinear",
        val_dataset=val_dataset,
        batch_size=128,
        num_epochs=10,
        save_every=5,
    )
    # trainer.load(step=5)
    trainer.train()

    model.eval()
    trainer.evaluate()