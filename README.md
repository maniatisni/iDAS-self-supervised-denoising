# Distributed Acoustic Sensing Self Supervised Denoising - PyTorch

Based on J-invariance denoising (Noise2Self)  
and original paper by Martijn van den Ende et. al. : 

*A Self-Supervised Deep Learning Approach for
Blind Denoising  
and Waveform Coherence
Enhancement in Distributed Acoustic Sensing data*.
     
This is my implementation, written in PyTorch (still in progress),  
for my MSc Thesis on the Data Science & Machine Learning Master Course. 

----------
----------

# Work Summary
## Raw Data Overview
Data collection was performed using a Silixa DAS Interrogator,  
utilizing a already deployed fibre optic cable, of length 44.5 km, originally intended for telecommunication use.

Total number of channels is 5568, gauge length is 8m, resulting in 44544 meters of cable.
Sampling rate is 1kHz, meaning a 30 sec file contains 30000 samples, 
so each raw file has a shape of (5568,30000).

As trying to depict a raw file isn't really helpful because of the noise,  
we filter our data through a bandpass filter, keeping only frequencies inside the range 1-10 Hz.
Here is a visualization of an M=6, on October 19, 2021, at 34.7005°N, 28.2493°E,  
showing 2 minutes of data (4 files) and only channels 1500 through 3000.

<img src="https://github.com/maniatisni/DAS-Denoising/blob/main/etc/1.png" width="800">

We can clearly see the P-wave, and the S-wave, respectively.

We can also visualize one channel only, to see the waveform of that specific point in space.  
As the fibre optic cable is very long, and it starts out at land, then goes into the sea,  
and then goes into land again, we usually focus on channels 1700 through 2300 out of the 5568,  
in order to avoid noise from land, surface noise from the point the cable enters the sea,  
and also avoid going too far as noise increases linearly the longer you go through the cable,  
which makes sense as backscatters tend to accumulate. 

Here is a visualization of channel 1700, showing the first 30 seconds of the previous picture: 

<img src="https://github.com/maniatisni/DAS-Denoising/blob/main/etc/2.png" width="600">

----------
----------

## Deep Learning Self Supervised Denoising
Our self supervised deep learning method to denoise the previous data,  
is actually a different implementation of the afforementioned paper,
which can be found in this [Link](https://eartharxiv.org/repository/view/2136/).

We won't go into detail as to how this J-Invariant method works here.  
Initially the idea was to implement this method on PyTorch, as it's originally written in Tensorflow,  
without pre-training in synthetic data.  

This method showed poor results as the majority of our data consists of noise,  
so there's a big imbalance between noise and events.  

That's when we started considering pre-training with synthetic data,  
and below we show the results of our first attempt, showing promise.

## Results
We are going to mention the basic steps we took to get these results.

### Pre-Processing
- The first step consists of **downsampling** our data, as 1kHz of sampling rate is too much, 
and the huge file sizes cause lots of data engineering problems, and also need more computing resources,  
without actually providing significant results than when downsampling.  
Final Sampling Rate was determined, as the sampling rate which would give off 2048 samples,  
from the initial 30000. This sampling rate is 68.27 Hz. 
     
- The second pre-processing step, is standardization, which basically means dividing our data with our standard deviation,
since the mean is almost always near zero.
     
- The third step, is filtering our data in a 1-10 Hz frequency band.
- The fourth and last step is selecting a subset of the spatial channels, specifically we chose channels 1700 through 2300 out of the 5568.

### Deep Learning Architecture
The deep learning architecture is based on the U-Net architecture.  We basically copied the architecture from the paper we mentioned before.
You can find the code for the model on this [link](https://github.com/maniatisni/DAS-Denoising/blob/main/model.py#L142) as it is on this repository.  
Since PyTorch doesn't offer the option to use anti-aliasing CNNs, or anti-aliasing Max Pooling, we used the layers from [adobe's implementation](https://github.com/adobe/antialiased-cnns).

### Synthetic Data
To better pretrain our network, we need to have some data that serves as a basic for our missing ground truth.
We took the synthetic data that were created by the paper's author.  

> *To gain a first-order understanding of the performance of the denoiser, we generate a synthetic data set with “clean” waveforms corrupted by Gaussian white noise with a controlled signal-to-noise ratio (SNR). The clean strain rate waveforms are obtained from three-component broadband seismometer recordings of the Pinon Flats Observatory Array,
California, USA, of 82 individual earthquakes. To simulate DAS strain rate recordings, we take two broadband stations in the array separated by a distance of 50 m and divide
the difference between their respective waveform recordingsby their distance. Owing to the low noise floor of these shallow borehole seismometers, the resulting strain rate waveforms exhibit an extremely high SNR.*

### Training Strategy
The idea is to pretrain on the synthetic data for 100 epochs, and then train on our real world data, 
for fewer epochs, for example 10, and see the results.

As this is a self supervised problem, and the aim is to actually denoise our data,  
there is nothing forbidding us to train on the data we want to denoise.  
Meaning it's okay to train on the "test-set". 

As training loss we used Mean Squared Error, and to properly evaluate our results we used the local waveform coherence CC around the k-th DAS channel:

![](https://latex.codecogs.com/png.latex?%5Cmathrm%7BCC%7D_%7Bk%7D%3D%5Cfrac%7B1%7D%7B4%20N%5E%7B2%7D%7D%5Cleft%5B%5Csum_%7Bi%2C%20j%3D-N%7D%5E%7B&plus;N%7D%20%5Cmax%20%5Cleft%28%5Cfrac%7Bx_%7Bk&plus;i%7D%20*%20x_%7Bk&plus;j%7D%7D%7B%5Csqrt%7B%5Csum_%7Bt%7D%20x_%7Bk&plus;i%7D%5E%7B2%7D%20%5Csum_%7Bt%7D%20x_%7Bk&plus;j%7D%5E%7B2%7D%7D%7D%5Cright%29-2%20N-1%5Cright%5D)

The coherence gain is then defined as the local coherence computed for the J-invariant reconstruction divided by that of the input data. As
such, coherence gains above 1 indicate that the reconstruction exhibits improved waveform coherence compared to the
input data, which is beneficial for coherence-based seismological analyses (template matching, beamforming).

### Preliminary Results
Results can be seen at the [testing notebook](https://github.com/maniatisni/DAS-Denoising/blob/main/testing.ipynb).
It's important to note, that the effect of denoising is more evident the bigger tha channel is, for example on the notebook it's more evident after channel 300.
Coherence Gain is still very low, we need to optimize the hyperparameters of this problem, but it's good to see that this method started making results. 

### Next Steps 
- Data Augmentation on synthetic data
- Hyperparameter optimization (Learning Rate, Number of Epochs, Number of Hidden Layers, both on synthetic data and on real DAS data)
- Experiment with data with smaller events
- Evaluate and Repeat
