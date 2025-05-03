import torch
import numpy as np

def generate_random_mask(batch_shape, mask_ratio=0.15):
    """
    Generate a binary mask tensor for masking random elements in the input.
    batch_shape: (B, C, T)
    Returns: mask of same shape with 0s indicating masked values
    """
    batch_size, channels, timesteps = batch_shape
    mask = torch.ones(batch_size, channels, timesteps)
    num_total = channels * timesteps
    num_mask = int(mask_ratio * num_total)
    for b in range(batch_size):
        idx = torch.randperm(num_total)[:num_mask]
        coords = np.unravel_index(idx.numpy(), (channels, timesteps))
        mask[b, coords[0], coords[1]] = 0
    return mask

def masked_mse_loss(y_pred, y_true, mask):
    """
    Computes masked MSE loss between predictions and targets.
    y_pred: (B, C)
    y_true: (B, C)
    mask: (B, C, T)
    """
    mask = (mask.sum(dim=2) > 0).float()
    loss = (mask * (y_pred - y_true)**2).sum() / (mask.sum() + 1e-6)
    return loss

def train_trace_masked_model(model, train_loader, num_epochs=20, lr=0.001, mask_ratio=0.15, device='cuda'):
    """
    Train TRACE-style encoder using masked self-supervised learning.
    Returns: trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        count_batches = 0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            mask = generate_random_mask(X_batch.shape, mask_ratio=mask_ratio).to(device)
            reconstructed, _ = model(X_batch, mask)
            X_target = X_batch.mean(dim=2)
            loss = masked_mse_loss(reconstructed, X_target, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count_batches += 1
        avg_loss = total_loss / count_batches
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}')

    print('Training finished!')
    return model

def train_transfer_classifier(model, loader, labels, num_epochs=20, lr=0.001, device='cuda'):
    """
    Train transfer classifier on top of frozen encoder.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    labels = torch.tensor(labels, dtype=torch.long)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        idx = 0

        for X_batch, _ in loader:
            batch_size = X_batch.size(0)
            y_batch = labels[idx:idx+batch_size]
            idx += batch_size

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%')

    print('Finished Transfer Training!')
