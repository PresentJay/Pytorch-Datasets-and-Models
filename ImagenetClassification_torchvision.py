# ref : https://csm-kr.tistory.com/m/6

from torchvision import models, datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

def main(device=None):
    IMAGENET_DIR = 'D:\datasets\ImageNet'
    BATCH_SIZE = 100
    NUM_WORKERS = 4
    
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]
    )
    
    val = datasets.ImageNet(root=IMAGENET_DIR, split='val', transform=transform)
    val_loader = data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    model = models.alexnet(pretrained=True).to(device)
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for index, (images, labels) in enumerate(val_loader):
            images = images.to(device)  # [BATCH_SIZE, 3, 224, 224]
            labels = labels.to(device)  # [BATCH_SIZE]
            outputs = model(images)
            
            # rank 1
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (pred == labels).sum().item()
            
            
            print("step : {} / {}".format(index+1, len(val)/int(labels.size(0))))
            print("top-1 percentage: {0:0.2f}%".format(correct_top1 / total * 100))
            
    
    print("top-1 percentage: {0:0.2f}%".format(correct_top1 / total * 100))    
            
        

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main(device)