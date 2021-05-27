from torchvision import models, transforms
import torch

def main():
    print(dir(models))
    UsingAlexNet()
   

def UsingAlexNet(test=True):
    # It is one of the early breakthrough networks in Image Recognition.
    
    # Step 1. Load the pretrained model
    
    alexnet = models.alexnet(pretrained=True) 
    # Downloading: "https://download.pytorch.org/models/alexnet-owt- 4df8aa71.pth"
    # to /home/hp/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth
    # Note that usually the PyTorch models have an extension of .pt or .pth
    
    # Once the weights have been downloaded, we can proceed with the other steps.
    if test:
        print(alexnet)
        
        """
        AlexNet(
            (features): Sequential(
                (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
                (1): ReLU(inplace=True)
                (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
                (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
                (4): ReLU(inplace=True)
                (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
                (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (7): ReLU(inplace=True)
                (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (9): ReLU(inplace=True)
                (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (11): ReLU(inplace=True)
                (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
            (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
            (classifier): Sequential(
                (0): Dropout(p=0.5, inplace=False)
                (1): Linear(in_features=9216, out_features=4096, bias=True)
                (2): ReLU(inplace=True)
                (3): Dropout(p=0.5, inplace=False)
                (4): Linear(in_features=4096, out_features=4096, bias=True)
                (5): ReLU(inplace=True)
                (6): Linear(in_features=4096, out_features=1000, bias=True)
            )
        )
        """
        
    # Step 2. Specify Image transformations
    
    # Once we have the model with us, the next step is to transform the input image
    # so that they have the right shape and other characteristics like mean and standard deviation.
    
    # These values should be similar to the ones which were used while training the model.
    # This make sure that the network will produce meaningful answers.
    
    # We can pre-process the input image with the help of transforms present in torchvision module.
    # In this case, we can use the following transforms for both AlexNet and ResNet.
    
    # transforms.Compose = combination of all the image transformations to be carried out on the input image.
    transform = transforms.Compose([
        transforms.Resize(256),             # Resize the image to 256x256 pixels.
        transforms.CenterCrop(224),         # Crop the image to 224x224 pixels about the center.
        transforms.ToTensor(),              # Convert the image to pyTorch Tensor data type.
        transforms.Normalize(               # Normalize the image by setting its mean and standard deviation.
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]    
        )
    ])
    
    
    # Step 3. Load the input image and pre-process it
    
    # Pillow module extensively with torchvision as it's the default image backend supported by torchvision.
    
    import PIL.Image
    img = PIL.Image.open('data/dog.jpg')
    img.show()
    
    # preprocess the image and prepare a batch to be passed through the network
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    
    # Step 4. Model Inference
    
    # we need to put out model in eval mode.
    alexnet.eval()
    
    # let's carry out the inference.
    out = alexnet(batch_t)
    print(out.shape)
    # torch.Size([1, 1000])
    
    # We still haven't got the class (or label) of the image.
    # For this, we will first read and store the labels from a text file having a list of all the 1000 labels.
    # Note that the line number specified the class number
    # so it's very important to make sure that you don't change that order.
    
    with open('data/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Since AlexNet and ResNet have been trained on the same ImageNet dataset
    # we can use the same classes list for both models.
    
    # Now, we need to find out the index where the maximum score in output vector out occurs.
    # We will use this index to find out the prediction
        
    _, index = torch.max(out, 1)
    
    percentage = torch.nn.fuctional.softmax(out, dim=1)[0] * 100
    
    print(classes[index[0]], percentage[index[0]].item())
    
    
    
if __name__ == '__main__':
    main()