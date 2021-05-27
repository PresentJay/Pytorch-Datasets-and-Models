# Reference https://learnopencv.com/pytorch-for-beginners-basics/

""" Pytorch : GPU Acceleration, Support for Distributed Computing and automatic gradient calculation """

""" Basic Pytorch Workflow (Overview of Pytorch Library) """

# Training Data (torch.utils.data) => Create Network (torch.nn)
# Create Network (torch.nn) => Train Network (torch.optim) (torch.autograd)
# Train Network (torch.optim) (torch.autograd) => Trained Model
# Testing data => Trained Model
# Trained Model => Get Prediction
# Trained Model => Model Interoperability (torch.onnx) (torch.jit)
# Trained Model => Publish Models (torch.hub)


""" 1. Data Loading and Handling """

# torch.utils.data => Dataset, DataLoader

# Dataset => Tensor data type based, mainly used for custom dataset
# DataLoader => Load datasets in background


""" 2. Building Neural Network """

# torch.nn  => for build neural network
#           => It provides all the common neural network layers
#           => such as fully connected layers, convolutional layers, activation and loss funtions etc.

# torch.optim   =>  training network (update weights and bias)

# torch.autograd    =>  for backward


""" 3. Model Inference and Compatibility """

# After the model has been trained, it can be used to predict output for test cased or even new datasets.
# This process is reffered to as model inference.

# Pytorch also provides TorchScript which can be used to run models independently from a Python runtime.
# This can be thought of as a Virtual Machine with instructions mainly specific to Tensors.

# You can also convert model trained using Pytorch into formats like ONNX,
# which allow you to use these models in other DL frameworks such as MXNet, CNTK, Caffe2.
# You can also convert onnx models to Tensorflow.

def main():
    import torch

    # Create a Tensor with just ones in a column
    a = torch.ones(5)

    # print the tensor we created
    print(a)
    # tensor([1., 1., 1., 1., 1.])
    
    # Create a Tensir with just zeros ina column
    b = torch.zeros(5)
    print(b)
    # tensor([0., 0., 0., 0., 0.])
    
    c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(c)
    # tensor([1., 2., 3., 4., 5.])
    
    d = torch.zeros(3,2)
    print(d)
    # tensor([[0., 0.],
    #     [0., 0.],
    #     [0., 0.]])
    
    e = torch.ones(3,2)
    print(e)
    # tensor([[1., 1.],
    #     [1., 1.],
    #     [1., 1.]])
    
    f = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f)
    # tensor([[1., 2.],
    #     [3., 4.]])
    
    
    # 3D Tensor
    g = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    print(g)
    # tensor([[[1., 2.],
    #      [3., 4.]],

    #     [[5., 6.],
    #      [7., 8.]]])

if __name__ == "__main__":
    main()