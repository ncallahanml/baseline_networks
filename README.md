## Simple 1D ANNs
This repository contains basic PyTorch neural networks with varying architectures that use synthetic sinusoidal input data. Includes a variety of dataloaders and PyTorch training additions, including [Lightning](https://lightning.ai/docs/pytorch/stable/) 
and [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) for reference. These are meant for my own reference when building more complex networks intended for real world scenarios, they don't include any domain specifics or complex implementations, nor do they focus on hyperparameter tuning or thorough demonstration. Every input is derivative of simple 1D sine waves precisely because this requirement allows all the hard parts of loading, storing and preprocessing data to be set aside in favor of debugging and implementing the network architectures quickly.

## Sine Wave Generation
#### PyTorch `RecursiveWaveGen`
- Recursive generator for sine waves with combinations of input hyperparameters
- Input parameters are lazily evaluated, meaning each parameter method call adds expected operations to the class until `.sample()` is called, causing the tensors to be instantiated and returned as a single tensor
- All computations are done in PyTorch, returned results are PyTorch Tensor instances as well

#### NumPy `WaveGen`
- Each method calls transforms the underlying array, defined by initial arguments and method setters
- After calling `.sample()`, array is resampled with noise a select number of times
- Once resampled, the `.samples` attribute can be used to access different versions of the array with additional noise
- After the underlying samples are handled, parameters can again be modified with method calls to make changes, the array can then be resampled, so on and so forth

#### General Operations
The following can be adjusted with method calls:
- Sine/Cosine phase
- Horizontal/Vertical flips
- Gaussian noise added
- Alteration of amplitude & bias
- Repeat of current array
- Custom adjustment of phase angle
- Alteration of number of periods

## Current Notebook Progress

Notebook Contents :
- Synthetic Generator : Sine wave generation for intended purpose is included
- Data Loader : Synthetic data is properly train/test split for model training
- Model Definition : PyTorch model is defined
- Training : Proper training & testing code is included
- Visualization : Visual evaluation of the input and model output is visible

Notebook Status : 
- Complete : Whole notebook works for the intended purpose
- Unrefined : At least one model has been tested, but the code needs to be cleaned up
- Untested : General code structure is finished, but training/visuals need to be tested
- In Progress : Enough code is available to view the intent, but the model is not ready for testing

| **File** | **Status** | **Description** | **Implementation Library** | **Extra** |
|:--------:|:----------:|:---------------:|:--------------------------:|:---------:|
| [Bayesian Regressor](BayesianRegressor.ipynb) | Unrefined | Probabilistic Linear Regression Model | PyTorch | Torch BNN Library |  
| [Binary Separator](BinaryDistributionSeparator.ipynb) | Complete | Time series segmentation intended to maximize distribution differences while being constrained by autocorrelation in each class | PyTorch | |  
| [CNN LSTM](CNNLSTM.ipynb) | Untested | Simple 1D CNN for feeding down to an LSTM forward predictor | PyTorch | Ray Tune |  
| [TCN](CNNRegressor.ipynb) | Complete | Time convolutional pooling down to linear model for regressing scalar value | PyTorch | |  
| [DAE](DAE.ipynb) | Complete | Denoising autoencoder for reconstructing sine waves from bottleneck | PyTorch | |  
| [Linear Ensemble](LinearEnsemble.ipynb) | Unrefined | Simple linear regressor with training for synthetic compositions of weights, allowing objective measure of coefficients against actual weights | PyTorch | |  
| [Siamese](Siamese.ipynb) | Unrefined | Siamese network for similarity learning between two sine waves | PyTorch | |  
| [Unet](Unet.ipynb) | Unrefined | Single dimensional UNet for supervised time series segmentation labeling | PyTorch | |  
| [Forward Ranker](ForwardRanker.ipynb) | Unrefined | Model that predicts forward time series rankings with limited parameters | PyTorch | | 
| [DDPM](DDPM.ipynb) | In Progress | Single dimensional denoising diffusion probabilistic model for generating new data | PyTorch |
| [Sharpe Regressor](PortfolioOptimizer.ipynb) | In Progress | Simple linear model optimizing for sharpe loss and alternative financial portfolio metrics as an alternative optimizer | Flax | |
| [Bidirectional Unet Encoding](UnetBiGRU.ipynb) | In Progress | Simple Unet encoder/decoder for extracting 3D features for Bidirectional GRU analysis | Flax | |  