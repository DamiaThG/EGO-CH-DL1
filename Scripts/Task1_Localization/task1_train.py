# Script di addestramento per la Room Localization (Task 1)
import os
import json
import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from Scripts.Task1_Localization.task1_dataset import get_dataloaders
from Scripts.Task1_Localization.task1_metrics import compute_ff1, compute_asf1
import numpy as np

class Task1LightningModule(L.LightningModule):

    def __init__(self, model, num_classes, lr=0.001, weight_decay=0.0001, max_epochs=50, label_smoothing=0.0):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
        self._val_preds = []
        self._val_labels = []
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x, lengths=None):
        return self.model(x, lengths)

    def training_step(self, batch, batch_idx):
        features, labels, mask, lengths = batch
        logits = self(features, lengths)
        B, T, C = logits.shape
        loss = self.loss_fn(logits.reshape(B * T, C), labels.reshape(B * T))
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels, mask, lengths = batch
        logits = self(features, lengths)
        B, T, C = logits.shape
        loss = self.loss_fn(logits.reshape(B * T, C), labels.reshape(B * T))
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, batch_size=B)
        preds = logits.argmax(dim=-1)
        for b in range(B):
            L_b = lengths[b].item()
            self._val_preds.append(preds[b, :L_b].cpu().numpy())
            self._val_labels.append(labels[b, :L_b].cpu().numpy())

    def on_validation_epoch_end(self):
        if len(self._val_preds) == 0:
            return
        ff1 = compute_ff1(self._val_preds, self._val_labels)
        asf1 = compute_asf1(self._val_preds, self._val_labels)
        self.log('val/FF1', ff1, prog_bar=True)
        self.log('val/ASF1', asf1, prog_bar=True)
        print(f'\n[Epoch {self.current_epoch}] val/FF1={ff1:.4f} | val/ASF1={asf1:.4f}')
        self._val_preds.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-06)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

def main():
    parser = argparse.ArgumentParser(description='Task 1 Training — Mamba Room Localizer')
    parser.add_argument('--features_dir', type=str, required=True, help='Path alla cartella delle feature (es. data/Task1_Features/Bellomo)')
    parser.add_argument('--val_features_dir', type=str, default=None, help='Path opzionale alla cartella delle feature di validation')
    parser.add_argument('--model', type=str, default='mamba', choices=['mamba'], help='Tipo di modello da usare')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing per CrossEntropyLoss (consigliato: 0.1)')
    parser.add_argument('--patience', type=int, default=15, help='Patience per EarlyStopping (consigliato: 20)')
    parser.add_argument('--output_dir', type=str, default='Scripts/Task1_Localization/experiments')
    parser.add_argument('--run_name', type=str, default='task1_mamba')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    os.makedirs(f'{args.output_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{args.output_dir}/logs', exist_ok=True)
    train_loader, val_loader, num_classes = get_dataloaders(args.features_dir, val_features_dir=args.val_features_dir, batch_size=args.batch_size, val_split=0.2, num_workers=args.num_workers)
    print(f'Classi: {num_classes} | Train: {len(train_loader.dataset)} seqs | Val: {len(val_loader.dataset)} seqs')
    from Scripts.Task1_Localization.models.task1_model_mamba import MambaRoomLocalizer
    model = MambaRoomLocalizer(input_dim=384, d_model=args.d_model, num_layers=args.num_layers, num_classes=num_classes, d_state=args.d_state, dropout=args.dropout)
    lit_model = Task1LightningModule(model=model, num_classes=num_classes, lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.epochs, label_smoothing=args.label_smoothing)
    wandb_logger = WandbLogger(project='EGO-CH-Task1', name=args.run_name, save_dir=args.output_dir, offline=True)
    checkpoint_cb = ModelCheckpoint(dirpath=f'{args.output_dir}/checkpoints', filename=f'{args.run_name}_best_{{val/ASF1:.4f}}', monitor='val/ASF1', mode='max', save_top_k=2)
    early_stop_cb = EarlyStopping(monitor='val/ASF1', mode='max', patience=args.patience, verbose=True)
    trainer = L.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=1, logger=wandb_logger, callbacks=[checkpoint_cb, early_stop_cb], enable_progress_bar=False, log_every_n_steps=10, precision='bf16-mixed', gradient_clip_val=1.0)
    print(f'\nAvvio training: {args.model.upper()} | {args.epochs} epoche | lr={args.lr}')
    trainer.fit(lit_model, train_loader, val_loader)
    print(f'\nTraining completato. Checkpoint migliore: {checkpoint_cb.best_model_path}')
if __name__ == '__main__':
    main()