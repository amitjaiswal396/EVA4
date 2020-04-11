import torch
from tqdm import tqdm

# model evaluation
def test(model, device, test_loader, criterion, test_losses, test_accuracy, misclassified_images):
    model.eval()
    test_loss, correct, incorrect_images = 0, 0, 0
    incorrect_img_dict = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            ### identifying misclassified images
            pred = pred.view_as(target) 
            mis_idx = pred != target
            incorrect_images = data[mis_idx]
            mis_target = target[mis_idx]
            mis_pred = pred[mis_idx]
            incorrect_img_dict = {'mis_img':incorrect_images, 'target_class':mis_target, 'predicted_class':mis_pred}
            misclassified_images.append(incorrect_img_dict)
            
            
    
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    total = len(test_loader.dataset)
    accuracy = 100. * correct / total
    test_accuracy.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(test_loss, correct, total, accuracy))
    return test_losses, test_accuracy, accuracy, misclassified_images
