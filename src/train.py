import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeechDataset, SpeechCollate
from model import TransformerTTS
from hparams import HParams

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hp = HParams()
    model = TransformerTTS(hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion_mel = nn.L1Loss()
    criterion_stop = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0).to(device), reduction='none')

    start_epoch = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Explicitly move optimizer states to device (Fixes extreme slowdown issue!)
        for state in optimizer.state.values():
             for k, v in state.items():
                 if isinstance(v, torch.Tensor):
                     state[k] = v.to(device)
                     
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming at epoch {start_epoch + 1}")


    # Load dataloader
    dataset = SpeechDataset(args.metadata_path)
    collate_fn = SpeechCollate()
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.overfit_mode, # Disable shuffle if overfitting
        collate_fn=collate_fn,
        drop_last=False
    )

    if args.overfit_mode:
        print("OVERFIT MODE: Training on a single batch to test network correctness...")
        single_batch = next(iter(loader))
        epochs = args.overfit_epochs
        
        import time
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            text_seqs = single_batch['text_seqs'].to(device)
            mels = single_batch['mels'].to(device)
            stop_tokens_target = single_batch['stop_tokens'].to(device)
            
            # Re-pad sequences by 1 so we don't drop the last audio frame on the right shift
            N_batch, n_mels, TIME = mels.shape
            pad_mel = torch.zeros(N_batch, n_mels, 1, device=device)
            mels = torch.cat([mels, pad_mel], dim=2)
            
            pad_stop = torch.ones(N_batch, 1, device=device)
            stop_tokens_target = torch.cat([stop_tokens_target, pad_stop], dim=1)
            
            target_mels = mels.transpose(1, 2) # (N, TIME+1, n_mels)

            # Autoregressive teacher forcing: right-shift inputs
            SOS = torch.zeros(N_batch, n_mels, 1, device=device)
            mels_input = torch.cat([SOS, mels[:, :, :-1]], dim=2)

            mel_postnet, mel_linear, stop_token_out = model(
                text=text_seqs,
                mel=mels_input,
                stop_tokens=stop_tokens_target
            )

            # Re-apply masking to targets exactly like model outputs
            bool_mel_mask = model.tgt_key_padding_mask.ne(0).unsqueeze(-1).expand_as(target_mels)
            target_mels_masked = target_mels.masked_fill(bool_mel_mask, 0.0)

            loss_linear = criterion_mel(mel_linear, target_mels_masked)
            loss_postnet = criterion_mel(mel_postnet, target_mels_masked)
            
            valid_mask = ~model.tgt_key_padding_mask.ne(0) # (N, TIME)
            loss_stop_unreduced = criterion_stop(stop_token_out, stop_tokens_target)
            loss_stop = (loss_stop_unreduced * valid_mask).sum() / (valid_mask.sum() + 1e-8)

            loss = loss_linear + loss_postnet + loss_stop
            
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Mel: {loss_postnet.item():.4f} | Stop: {loss_stop.item():.4f} | Time: {elapsed:.2f}s")
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"model_overfit_{timestamp}.pth")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, save_path)
        print(f"Overfitting test completed. Check if loss approached 0. Model saved to {save_path}")
        return

    # Normal training mode
    print(f"Starting standard training for {args.epochs} epochs...")
    import time
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            text_seqs = batch['text_seqs'].to(device)
            mels = batch['mels'].to(device)
            stop_tokens_target = batch['stop_tokens'].to(device)
            
            # Re-pad sequences by 1 so we don't drop the last audio frame on the right shift
            N_batch, n_mels, TIME = mels.shape
            pad_mel = torch.zeros(N_batch, n_mels, 1, device=device)
            mels = torch.cat([mels, pad_mel], dim=2)
            
            pad_stop = torch.ones(N_batch, 1, device=device)
            stop_tokens_target = torch.cat([stop_tokens_target, pad_stop], dim=1)
            
            target_mels = mels.transpose(1, 2)

            # Autoregressive teacher forcing: right-shift inputs
            SOS = torch.zeros(N_batch, n_mels, 1, device=device)
            mels_input = torch.cat([SOS, mels[:, :, :-1]], dim=2)

            mel_postnet, mel_linear, stop_token_out = model(
                text=text_seqs,
                mel=mels_input,
                stop_tokens=stop_tokens_target
            )

            bool_mel_mask = model.tgt_key_padding_mask.ne(0).unsqueeze(-1).expand_as(target_mels)
            target_mels_masked = target_mels.masked_fill(bool_mel_mask, 0.0)

            loss_linear = criterion_mel(mel_linear, target_mels_masked)
            loss_postnet = criterion_mel(mel_postnet, target_mels_masked)
            
            valid_mask = ~model.tgt_key_padding_mask.ne(0) # (N, TIME)
            loss_stop_unreduced = criterion_stop(stop_token_out, stop_tokens_target)
            loss_stop = (loss_stop_unreduced * valid_mask).sum() / (valid_mask.sum() + 1e-8)

            loss = loss_linear + loss_postnet + loss_stop
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'mel': f"{loss_postnet.item():.4f}",
                'stp': f"{loss_stop.item():.4f}"
            })
            
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | Total Time: {elapsed:.2f}s")
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(args.save_dir, exist_ok=True)
            
            if (epoch + 1) == args.epochs:
                save_path = os.path.join(args.save_dir, f"model_final_epoch_{epoch+1}_{timestamp}.pth")
            else:
                save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}_{timestamp}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mini SpeedySpeech Autoregressive Model")
    parser.add_argument("--metadata_path", type=str, default="data/processed/processed_metadata.csv", help="Path to preprocessed metadata CSV.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--overfit_mode", action="store_true", help="Train on a single batch to test network correctness (overfitting test).")
    parser.add_argument("--overfit_epochs", type=int, default=700, help="Number of epochs for single batch overfit mode.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save the model.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from.")
    
    args = parser.parse_args()
    train(args)
