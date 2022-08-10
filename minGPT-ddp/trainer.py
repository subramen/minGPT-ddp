"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from dataclasses import dataclass, asdict
from collections import OrderedDict
from typing import Optional, Any, Dict
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


@dataclass
class TrainerConfig:
    # job_name: str
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    # enable_profile: bool = False
    # log_dir: Optional[str] = None

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int

class Trainer:

    def __init__(self, trainer_config, model, optimizer, train_dataset, test_dataset=None):
        self.config = trainer_config
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])  
        # data stuff
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None
        # initialize train states
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer        
        self.save_every = self.config.save_every
        # load snapshot if available
        self._load_snapshot(self.config.snapshot_path)
        # wrap with DDP
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset)
        )

    def _load_snapshot(self, snapshot_path):
        # optionally pass cloud object store url to snapshot_path
        if os.path.exists(snapshot_path):
            snapshot_data = torch.load(snapshot_path, map_location="cpu")
            snapshot = Snapshot(**snapshot_data)
            self.model.load_state_dict(snapshot.model_state)
            self.optimizer.load_state_dict(snapshot.optimizer_state)
            self.epochs_run = snapshot.finished_epoch
            print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        else:
            print("No snapshot file found! Starting from scratch")

    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train):
            _, loss = self.model(source, targets)
        
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()
        
        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        for iter, (source, targets) in enumerate(dataloader):
            step_type = "Train" if train else "Test"
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets, train)
            if iter % 100 == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch+1} | Iter {iter} | {step_type} Loss {batch_loss:.5f}")

    def _save_snapshot(self, epoch):
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch
        )
        torch.save(asdict(snapshot), "snapshot.pt")
        # optionally upload to the cloud
        print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch, self.train_loader, train=True)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            # eval run
            if self.test_loader:
                self._run_epoch(epoch, self.test_loader, train=False)

