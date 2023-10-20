# Goal
Build basic neural networks with varying architectures that use sinusoidal input data. Includes a variety of dataloaders and PyTorch training additions, including [Lightning]() 
and [Tensorboard]() for reference. Using sinusoidal data with noise should prove the simplest version of viability for the training procedure while not requiring much training time, allowing complexity for real data and higher dimensional inputs to be built out later.

# Current Progress
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


# Variations in Synthetic Data
- **Sine Wave**:  
    - `period` : period of each sign wave, target variable  
    - `amplitude` : amplitude associated with the sine wave  
    - `phase` : where the sine wave starts horizontally
    - `offset` : y location of the sine wave
- **Noise**:
    - `noise distribution` : generating distribution for the noise added to the wave
    - `noise magnitude` : magnitude of noise added to the sine wave
    - `noise method` : multiplicative or additive noise
- **Additional**:
    - `padding` : how values are padded, primarily concerning if sine waves are repeated for multiple periods or only a single period with padding values
    - `slant` : 
    
# Eventual Supervision Labels

- Sine/Flat : int
- Sine/Flat/Noisy : int
- % Sine vs Flat : float
- Denoised : array_like
- Period/Parameters : float/array_like
- Sine Indices : array_like

# Model Goals
- Latent Representation
- Denoising
- Distribution Separation
- Probabilistic Prediction
