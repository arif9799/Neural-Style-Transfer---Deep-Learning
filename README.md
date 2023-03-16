# Neural-Style-Transfer---Deep-Learning


## **Style Transfer.**
- Style Transfer is the task of imposing a style onto a content image. 
To elaborate more, Given two images, first one containing the content and second one containing an artistic style, 
the task is to produce an image with the exact contents of the content image along with some amount of style injected from the style image. 
- Looking at 2 illustrative examples below:
  - First image portrays a well known American Entrepreneur & the most notable Innovator Steve Jobs - *"Content Image"*
  - Second image corresponds to a random artistic style - *"Style Image"*
  - Third image represents the task of Style Transfer, imposing the style onto the contents - *"Output Image"*
    - ![NST1](https://user-images.githubusercontent.com/93501171/225389414-e050e6b4-13ac-4f2f-8a81-6819c057777f.png)
    - ![NST2](https://user-images.githubusercontent.com/93501171/225390189-e5eef6de-df82-45ba-8daa-47a9bd659e2d.png)

## **Neural Style Transfer.**
- Neural Style Transfer is the domain which uses pretrained Deep Neural Networks to accomplish the task of Style Transfer. It does not require training
but uses Convolutional Neural Networks as a back-bone to achieve desired results. It is an algorithmic technique that takes 3 Images as input, extracts low level features, calculates loss amongst them, backpropagates losses to the image to be generated (the pretrained weights are frozen and not altered during backpropagation) and produces the output. *(Should make more sense in the Working Mechanism below)*

- The three inputs are:
  - Content Image
  - Style Image
  - Noise Image (Initial Image that will contain Contents with Style imposed onto it later on)
    - (Can be randomly generated using Gaussian Distribution or Content Image can be assigned to it initially)
  
- It starts with extracting low level representations or feature maps of different convolutional layers, calculates the content loss from the feature maps of deepest convolutional layer & style loss from the gram matrices constructed from the feature maps of all convolution layers & the weighted cumulative loss *(Sum of Content and Style loss)* is backpropagated to the Noise Image.

- Content Loss is mean of sum of squared differences between the feature map (from the deepest layer) of Content Image & the Noise, multiplied by a factor ***'Alpha'*** that determines how much of the loss is to be backpropagated from the Content Image

- Style loss is Summation of Mean of Sum of Squared Differences between the Gram Matrix of Style Image & the Noise for all convolutional layers, multiplied by a factor ***'Beta'*** that determines how much of the loss is to be backpropagated from Style Image
  - In a particular convolutional layer, Gram Matrix is the inner-product of flattened feature maps that represents correlation amongst themselves
  - For all convolution layers Gram Matrices are calculated, multiplied with a constant *(different for every convolutional layer)*

- The *Working Mechanism* of the algorithmic technique is as follows:

  - Step 1: Extract the Content & Noise Features of the deepest Convolution Layer & Calculate the Sum of Squared Differences between the Content Feature Maps & Noise Feature Maps multiply by the coeffecient *Alpha* -
  
    <!---------------------------------------------------------------------------------------------------------------------------------------------->
    <p align="center"> 
        <img height="240" alt="Picture1" src="https://user-images.githubusercontent.com/93501171/225460411-b557ded7-9073-4cc8-8576-fd6f797a7b9b.png">
      </p> 
    <!-- ![NST-Page-1 drawio](https://user-images.githubusercontent.com/93501171/225460411-b557ded7-9073-4cc8-8576-fd6f797a7b9b.png) -->
    
    <!---------------------------------------------------------------------------------------------------------------------------------------------->
    <p align="center"> 
        <img height="240" alt="Picture1" src="https://user-images.githubusercontent.com/93501171/225460738-3b0639e9-60e9-44b0-9c5c-415513229e81.png">
      </p> 
    <!-- ![NST-Page-2 drawio](https://user-images.githubusercontent.com/93501171/225460738-3b0639e9-60e9-44b0-9c5c-415513229e81.png) -->
    
    <!---------------------------------------------------------------------------------------------------------------------------------------------->
    <p align="center"> 
        <img height="120" alt="Picture1" src="https://user-images.githubusercontent.com/93501171/225461148-3fcf998d-a673-425b-a5c5-38dc394457b1.png">
      </p> 
    <!-- ![NST-Page-3 drawio](https://user-images.githubusercontent.com/93501171/225461148-3fcf998d-a673-425b-a5c5-38dc394457b1.png) -->
    
      

     
      
      
  - Step 2: Extract the Style & Noise Features, Construct Gram Matrices of all Convolutional Layers, Calculate the Sum of Squared Differences between the Gram Matrices of Style & Noise Image, Average the losses from all the layers, multiply by the coeffecient *Beta*
  
  
    <!---------------------------------------------------------------------------------------------------------------------------------------------->
    <p align="center"> 
        <img height="240" alt="Picture1" src="https://user-images.githubusercontent.com/93501171/225461390-bd4c2423-5f29-400c-b209-6323a917d10d.png">
      </p> 
    <!-- ![NST-Page-4 drawio](https://user-images.githubusercontent.com/93501171/225461390-bd4c2423-5f29-400c-b209-6323a917d10d.png) -->

    
    <!---------------------------------------------------------------------------------------------------------------------------------------------->
    <p align="center"> 
        <img height="240" alt="Picture1" src="https://user-images.githubusercontent.com/93501171/225461493-6073d5da-7b21-4f0e-ae6c-e41de2d44098.png">
      </p> 
    <!--  ![NST-Page-5 drawio](https://user-images.githubusercontent.com/93501171/225461493-6073d5da-7b21-4f0e-ae6c-e41de2d44098.png) -->

    
    <!---------------------------------------------------------------------------------------------------------------------------------------------->
    <p align="center"> 
        <img height="120" alt="Picture1" src="https://user-images.githubusercontent.com/93501171/225461533-22ca71cb-5767-4298-8e24-7c9d4d99d6d2.png">
      </p> 
    <!-- ![NST-Page-6 drawio](https://user-images.githubusercontent.com/93501171/225461533-22ca71cb-5767-4298-8e24-7c9d4d99d6d2.png) -->
    
  - Step 3: Backpropagate the Summed Loss to Noise Image (that could be Gaussian or Content Image itself)
  
    <!---------------------------------------------------------------------------------------------------------------------------------------------->
    <p align="center"> 
        <img height="480" alt="Picture1" src="https://user-images.githubusercontent.com/93501171/225462169-a000095b-887d-43d4-be19-e0b82a3fc00f.png">
    </p> 
    <!-- ![NST-Page-7 drawio](https://user-images.githubusercontent.com/93501171/225462169-a000095b-887d-43d4-be19-e0b82a3fc00f.png) -->
    
    
  - Step 4: Repeat Step 1, 2 and 3 for k epochs until the Noise Image contains the content along with some style injected into it.


## **Code Description**

  - The Code is mainly divided into 5 parts

#### Loading a Pretrained Convolutional Neural Network

#### Preprocessing Image

#### Deprocessing Image

#### Content & Style Loss Function

#### Driver Code 
  - Hyperparameter Tuning:
    - In the cell blocks of driver code that executes the whole mechanisn of Neural Style Transfer, for different values of *Alpha* and *beta* along variations in *learning rates*, you can see same results with different amounts or proportions of styles and contents injected into the results.


##### Lastly, an appealing convergence



https://user-images.githubusercontent.com/93501171/225492691-109bb503-8cba-4305-b33e-27bafa560831.mp4





  
