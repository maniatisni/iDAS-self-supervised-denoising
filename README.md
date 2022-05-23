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
# Problem with Santorini data summarized
### PyTorch
I started replicating the paper, as close to that as possible.  
It is evident in the [dataloaders.py](https://github.com/maniatisni/DAS-Denoising/blob/main/dataloaders.py) that i made the original dataloaders
from Tensorflow work in PyTorch.  

In [train-synthetic.ipynb](https://github.com/maniatisni/DAS-Denoising/blob/main/train-synthetic.ipynb) I train synthetic data
for the same number of epochs, hyperparameters e.t.c. as the paper, and in [testing-synthetic.ipynb](https://github.com/maniatisni/DAS-Denoising/blob/main/testing-synthetic.ipynb) it is evident that my code works as I get the same results as the original paper, except from the $R^2$ vs slowness,  
I will come back to this as I will retrain maybe for more epochs.  

Next, I started training in real DAS data, from HCMR/Nestor, in INSERT train-DAS-paper.ipynb
to replicate the same results as the paper, in order to be able to use these results as benchmarks.
The same results are evident in [testing-DAS-paper.ipynb](https://github.com/maniatisni/DAS-Denoising/blob/main/testing-DAS-paper.ipynb).

Finally, I wanted to used the pretrained model that was trained on synthetic data, to retrain Santorini DAS data, [train-DAS-santorini.ipynb](https://github.com/maniatisni/DAS-Denoising/blob/main/train-DAS-santorini.ipynb), for 200 epochs, as the same (not satisfying) results 
were given even for 50 epochs, as is evident in [testing-DAS-santorini.ipynb](https://github.com/maniatisni/DAS-Denoising/blob/main/testing-DAS-santorini.ipynb).
The model simply outputs the same as the input, only with a few differences in amplitude.  

Preprocessing for Santorini data is in the [preprocessing.py](https://github.com/maniatisni/DAS-Denoising/blob/main/preprocessing.py), 
if you ignore the 'if-elif' code where I just hardcoded the event reading and extracting so that the waveforms are centered approximately at the arrival of the first wave.  
The steps I took are:
- Read 2 files of 30 sec each, and keep only 41 seconds of recording
- Keep only 1000 "clean" channels out of the 5568 total channels
- Bandpass filter frequencies in the 1-10 Hz range
- Downsample, from 1kHz (41000 samples) to 50 Hz (2048 samples)
- Save as a new file

I also tried extracting the data for longer periods of time, 82 seconds of recording in 4096 samples, where I selected channels
in the range 1700-3700 but skipping one channel at a time, so for a final of 1000 channels, with the same results.

Normalization of waveforms (divide each channel with its standard deviation) happens before training/testing.

### Tensorflow
In order to gain some understanding on what may be on fault, I tried re-training the Santorini data,
using the [jDAS module](https://github.com/martijnende/jDAS) on Tensorflow, this can be seen on [jDAS-santorini.ipynb](https://github.com/maniatisni/DAS-Denoising/blob/main/jDAS-santorini.ipynb).  
Pretrained synthetic model taken from [here](https://figshare.com/articles/software/A_Self-Supervised_Deep_Learning_Approach_for_Blind_Denoising_and_Waveform_Coherence_Enhancement_in_Distributed_Acoustic_Sensing_data/14152277).
The fact that not even this module can denoise the data, tells me that something is wrong with the data in the first place.

**The only way I got satisfying denoising results, is if I used the synthetic-pretrained model, both on Tensorflow and on PyTorch.
This way the model isn't trained at the Santorini Data at all.**
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

That's when we started considering pre-training with synthetic data.

### Pre-Processing
- As our files are separated after 30 seconds of recording, sometimes some events are caught between two separate files,  
so we hand picked a few events, and tried to keep the data both before the event and during, and sometimes after,  
depending on the duration of each event.
Each training file, consists of 41 seconds of data recording.
- We then filter our data in a 1-10 Hz bandpass fiter. This means we keep frequencies only in this range.
- We **downsample** our data, as 1kHz of sampling rate is too much, 
     and the huge file sizes cause lots of data engineering problems, and also need more computing resources,
     without actually providing significant results than when downsampling. 
     Since our initial sampling rate is 1kHz, one event is 41000 samples. We downsample in the frequency 
     that results when the final sample number is 2048, which is ~50 Hz (49.95 Hz).

- The next step, is standardization, which basically means dividing our data with our standard deviation,
since the mean is almost always near zero.
     
- The last step consists of selecting a subset of the spatial channels, specifically we chose channels 1700 through 2700 out of the 5568, 
     so as the cable to be at full sea depth and not too far so as to allow the noise to accumulate.
     We split into train and validation sets on the channel axis (800 and 200 respectively).

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
We use these data to create waveforms that simulate real DAS recordings, showing various SNRs, that are also augmented.  
Fore details read the paper.

### Training Strategy
The idea is to pretrain on the synthetic data till convergence, for 2000 epochs, and then train on our real world data, 
for fewer epochs, for example 20, and see the results.

As this is a self supervised problem, and the aim is to actually denoise our data,  
there is nothing forbidding us to train on the data we want to denoise.  
Meaning it's okay to train on the "test-set". 

As training loss we used Mean Squared Error, and to properly evaluate our results we used the local waveform coherence CC around the k-th DAS channel:

![](https://latex.codecogs.com/png.latex?%5Cmathrm%7BCC%7D_%7Bk%7D%3D%5Cfrac%7B1%7D%7B4%20N%5E%7B2%7D%7D%5Cleft%5B%5Csum_%7Bi%2C%20j%3D-N%7D%5E%7B&plus;N%7D%20%5Cmax%20%5Cleft%28%5Cfrac%7Bx_%7Bk&plus;i%7D%20*%20x_%7Bk&plus;j%7D%7D%7B%5Csqrt%7B%5Csum_%7Bt%7D%20x_%7Bk&plus;i%7D%5E%7B2%7D%20%5Csum_%7Bt%7D%20x_%7Bk&plus;j%7D%5E%7B2%7D%7D%7D%5Cright%29-2%20N-1%5Cright%5D)

The coherence gain is then defined as the local coherence computed for the J-invariant reconstruction divided by that of the input data. As
such, coherence gains above 1 indicate that the reconstruction exhibits improved waveform coherence compared to the
input data, which is beneficial for coherence-based seismological analyses (template matching, beamforming).

