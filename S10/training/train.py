import torch
from tqdm import tqdm

# model training
def train(model, device, train_loader, optimizer, epoch, criterion, train_losses, train_accuracy):
    model.train()
    pbar = tqdm(train_loader)
    correct, processed, avg_loss = 0, 0, 0
    train_acc = []
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        max_prob = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += max_prob.eq(target.view_as(max_prob)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'epoch={epoch} Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
        avg_loss += loss.item()
        
    avg_loss /= len(train_loader.dataset)
    train_losses.append(avg_loss)
    total = len(train_loader.dataset)
    accuracy = 100. * correct / total
    train_accuracy.append(accuracy)
    return train_losses, train_accuracy