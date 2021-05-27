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
    
    
    print(f'{f.shape}, {e.shape}, {g.shape}')
    # torch.Size([2, 2]), torch.Size([3, 2]), torch.Size([2, 2, 2])
    
    
    # Access an element in Tensor
    
    # Get element at index 2
    print(c[2])
    # tensor(3.)
    
    # All indices starting from 0
    
    # Get element at row 1, column 0
    print(f[1, 0])
    # tensor(3.)
    
    # We can also use the following
    print(f[1][0])
    # tensor(3.)
    
    # Similarly for 3D Tensor
    print(g[1,0,0])
    # tensor(5.)
    
    print(g[1][0][0])
    # tensor(5.)
    
    
    # All elements
    print(f[:])
    # tensor([[1., 2.],
    #     [3., 4.]])
    
    # All elements from index 1 to 2 (inclusive)
    print(c[1:3])
    # tensor([2., 3.])
    
    # All elements till index 4 (exclusive)
    print(c[:4])
    # tensor([1., 2., 3., 4.])
    
    # First row
    print(f[0, :])
    # tensor([1., 2.])
    
    # Second Column
    print(f[:, 1])
    # tensor([2., 4.])
    
    
    # Specify data type of elements
    
    int_tensor = torch.tensor([[1,2,3,] ,[4,5,6]])
    print(int_tensor.dtype)
    # torch.int64
    
    # What if we changed any one element to floating point number?
    int_tensor = torch.tensor([[1,2,3], [4., 5,6]])
    print(int_tensor.dtype)
    # torch.float32
    
    # This can be overriden as follows
    int_tensor = torch.tensor([[1,2,3], [4.,5,6]], dtype=torch.int32)
    print(int_tensor.dtype)
    # torch.int32
    
    print(int_tensor)
    # tensor([[1, 2, 3],
    #     [4, 5, 6]], dtype=torch.int32)
    
    
    # Tensor to/from Numpy Array
    
    import numpy as np
    
    # Tensor to Array
    f_numpy = f.numpy()
    print(f_numpy)
    
    #  [[1. 2.]
    #  [3. 4.]]
    
    # Array to Tensor
    h = np.array([[8,7,6,5], [4,3,2,1]])
    h_tensor = torch.from_numpy(h)
    print(h_tensor)
    # tensor([[8, 7, 6, 5],
    #     [4, 3, 2, 1]], dtype=torch.int32)
    
    
    # Arithmetic Operations on Tensors
    
    # Create Tensor
    tensor1 = torch.tensor([[1,2,3], [4,5,6]])
    tensor2 = torch.tensor([[-1, 2, -3], [4,-5,6]])
    
    # Addition
    print(tensor1 + tensor2)
    # We can also use
    print(torch.add(tensor1, tensor2))
    # tensor([[ 0,  4,  0],
    #     [ 8,  0, 12]])
    
    # Substraction
    print(tensor1-tensor2)
    # We can also use
    print(torch.sub(tensor1, tensor2))
    # tensor([[ 2,  0,  6],
    #     [ 0, 10,  0]])
    
    # Multiplication
    print(tensor1 * 2)
    # tensor([[ 2,  4,  6],
    #     [ 8, 10, 12]])
    
    # Tensor with another tensor
    # Elementwise Multiplication
    print(tensor1 * tensor2)
    # tensor([[ -1,   4,  -9],
    #     [ 16, -25,  36]])
    
    # Matrix multiplication
    tensor3 = torch.tensor([[1,2] ,[3,4], [5,6]])
    print(torch.mm(tensor1, tensor3))
    # tensor([[22, 28],
    #     [49, 64]])
    
    # Division
    # Tensor with scalar
    print(tensor1/2)
    
    
    # Tensor with another tensor
    # Elementwise division
    print(tensor1/tensor2)
    # tensor([[-1.,  1., -1.],
    #     [ 1., -1.,  1.]])
    
    
    
    

if __name__ == "__main__":
    main()