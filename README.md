## Simple 1D ANNs
Build basic neural networks with varying architectures that use sinusoidal input data. Includes a variety of dataloaders and PyTorch training additions, including [Lightning](https://lightning.ai/docs/pytorch/stable/) 
and [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) for reference. Using sinusoidal data with noise should prove the simplest version of viability for the training procedure while not requiring much training time, allowing complexity for real data and higher dimensional inputs to be built out later.

## Sine Wave Generation
#### PyTorch `RecursiveWaveGen`
- Recursive generator for sine waves with combinations of input hyperparameters
- Input parameters are lazily evaluated, meaning each parameter method call adds expected operations to the class until `.sample()` is called, causing the tensors to be instantiated and returned as a single tensor
- All computations are done in PyTorch, returned results are a PyTorch tensor

#### NumPy `WaveGen`
- Each method call transforms the underlying array, defined by initial arguments and method setters
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

## Current Model Progress
| **Model** | **File** | **Status** | **Description** |
|:---------:|:--------:|:----------:|:---------------:|
| Bayesian Regressor | [SineBayesianRegressor](SineBayesianRegressor.ipynb) | Complete, Unrefined | Probabilistic Linear Regression Model |
| Binary Separator | [SineBinarySeparator](SineBinarySeparator.ipynb) | Complete | Time series segmentation intended to maximize distribution differences while being constrained by autocorrelation in each class |
| CNN LSTM | [SineCNNLSTM](SineCNNLSTM.ipynb) | Incomplete | Simple 1D CNN for feeding down to an LSTM forward predictor |
| CNN Regressor | [SineCNNRegressor](SineCNNRegressor.ipynb) | Complete | Time convolutional pooling down to linear model for regressing scalar value |
| 1D Denoising Autoencoder | [SineDAE](SineDAE.ipynb) | Complete | Denoising autoencoder for reconstructing sine waves from bottleneck |
| Linear Decomposer | [SineLinearComposition](SineLinearComposition.ipynb) | Complete, Limited | Simple linear regressor with training for synthetic compositions of weights, allowing objective measure of coefficients against actual weights |
| Siamese Network | [SineSiamese](SineSiamese.ipynb) | Complete, Unrefined | Siamese network for similarity learning between two sine waves |
| 1D UNet | [SineUnet](SineUnet.ipynb) | Incomplete | Single dimensional UNet for supervised time series segmentation labeling |
