## Introduction

<img align="left" width="360" height="175" src="https://github.com/maniatisni/DAS-Denoising/blob/main/etc/logo.png">
This is the code repository regarding my MSc Thesis titled:  
“*Denoising Distributed Acoustic Sensing data with Self-Supervised Deep Learning*”
Based on the concept self-supervised J-invariant denoising applied in Distributed Acoustic Sensing (seismic) data, heavily based on:

- *“A Self-Supervised Deep Learning Approach for Blind Denoising and Waveform Coherence Enhancement in Distributed Acoustic Sensing Data.” 2021. Software. Figshare. figshare. March 3, 2021. https://doi.org/10.6084/m9.figshare.14152277.v1.* [1]
- *Batson, Joshua, and Loic Royer. n.d. “Noise2Self: Blind Denoising by Self-Supervision,” 10.* [2]

----------
### Abstract
Distributed Acoustic Sensing (DAS) is an emerging technology utilizing fiber optic cables for vibration measurements with various applications such as seismic signal analysis, pipeline monitoring, traffic monitoring (roads, railways and trains). Its ease of use and versatility, lie on the fact that it can be deployed in harsh and dangerous environments, such as submarine, glaciated or volcanic and due to its ability to turn existing commercial fiber optic cables into sensor arrays with temporal sampling of up to 1 thousand samples per second, and with a spatial sampling in the order of meters. However, new environments also come with new challenges as each new environment has the ability to introduce noise in various types, lowering the quality of the measurements and thus impeding with the data analysis workflows. In this work, we explore the possibility of removing incoherent noise from DAS recordings, utilizing the concept of J-invariance and modern self-supervised deep learning methods, without making assumptions regarding the noise characteristics. We apply this method to both synthetic and real world DAS data, from four different experiments, one of which took place in a volcanic environment in Iceland, and the rest come from three separate submarine DAS recordings in Greece. The results show exceptional denoising capability and great promise to be incorporated into seismological analysis data workflows, when the noise is incoherent.

----------
### Results

#### Two events from Santorini DAS, that were catalogued:

|Origin Time (GMT)|Latitude|Longitude|Depth (km)|Magnitude|Location|Distance (km) from center|
|----|----|----|----|----|----|----|
|2021/10/19 20:08:12.770|359.573|255.650|14.0|2.9|52.4 km SSE of Thira|51.8|
|2021/10/19 20:41:54.270|359.423|255.770|7.0|2.6|54.3 km SSE of Thira|53.7|
<img align = "center" src="https://github.com/maniatisni/DAS-Denoising/blob/main/etc/santorini-events-3%264.jpeg" width="1500">
<img align = "center" src="https://github.com/maniatisni/DAS-Denoising/blob/main/etc/Santorini-wiggle-event-3.jpeg" width="1500">

#### Another application on Iceland data:
<img align = "center" src="https://github.com/maniatisni/iDAS-self-supervised-denoising/blob/main/etc/iceland-events-6%267.png" width="1500">
<img align = "center" src="https://github.com/maniatisni/iDAS-self-supervised-denoising/blob/main/etc/iceland-wiggle-event-6.png" width="1500">


---------
### Data
The experiment took place in Santorini Island, Greece, for a period of 2 months, though the data we used were mainly from the 19th and the 20th of October, 2021.
Data collection was performed using a standard Silixa DAS Interrogator, utilizing an already deployed fibre optic cable, 44.5 km in length, originally intended for telecommunication use. The fiber optic cable starts at Fira town, the capital of Santorini, and after a few kilometers enters the sea where it goes really close to the Kolumbo volcano, which is of great scientific interest, and then ends up on the mainland of Ios Island. 

The fiber optic cable is characterized by gauge length of 8m, which is equal to the channel spacing. The total number of channels is 5568, resulting in 44544 meters of cable, as we mentioned before.
Regarding the temporal sampling, sampling rate was selected at 1kHz, meaning 1 second of recording equals 1000 samples. This is a really high sampling rate, compared to other Distributed Acoustic Sensing experiments, thus resulting to a huge number of data generated.
A new file is generated after 30 seconds of recording, resulting in a matrix containing 30,000 samples of 5568 channels, at around 30 MB on average.

As trying to depict a raw file isn't really helpful because of the noise, we filter our data through a bandpass filter, keeping only frequencies inside the range 1-10 Hz.
Here is a visualization of an M=6, on October 19, 2021, at 34.7005°N, 28.2493°E,  howing 2 minutes of data (4 files) and only channels 1500 through 3000. We can clearly see the P-wave, and the S-wave, respectively.



<img align = "left" src="https://github.com/maniatisni/DAS-Denoising/blob/main/etc/1.png" width="499">
<img align="right" width="470" src="https://github.com/maniatisni/DAS-Denoising/blob/main/etc/2.png">



We can also visualize one channel only, to see the waveform of that specific point in space. As the fibre optic cable is very long, and it starts out at land, then goes into the sea, and then goes into land again, we usually focus on channels 1700 through 2300 out of the 5568, in order to avoid noise from land, surface noise from the point the cable enters the sea, and also avoid going too far as noise increases linearly the longer you go through the cable, which happens because backscatterers tend to accumulate. Here is a visualization of channel 1700, showing the first 30 seconds of the previous picture: 

----------

### Pre-Processing of DAS data
- As our files are separated after 30 seconds of recording, sometimes some events are caught between two separate files, so we extracted the events keeping 41 seconds of recording and centering the waveforms around the arrival of the first wave.
- We filter our data in a 1-10 Hz bandpass fiter.
- We **downsample** the data. Since our initial sampling rate is 1kHz, one event is 41000 samples. We downsample in the frequency that results when the final sample number is 2048, which conventiently is ~50 Hz (49.95 Hz).
- We select a subset of the spatial channels, specifically we chose channels 1700 through 2700 out of the 5568, so as the cable to be at full sea depth and not too far so as to allow the noise to accumulate.
- Before training and testing the data, we standardize the data by dividing each channel by the standard deviation.      

---------

### Generation and Preprocessing of Synthetic Data
If we start training on raw DAS data, there is a great imbalance between noise and clean "events". Due to that, the neural network will never be able to denoise the data successfully due to the missing ground truth. In order to do that, we prepare synthetic data taken from traditional borehole seismometers, where we can calculate the strain rate and also make small shifts by the gauge length in order to simulate DAS recordings. The clean strain rate waveforms are obtained from three-component broadband seismometer recordings of the Pinon Flats Observatory Array, California, USA, of 82 individual earthquakes. To simulate DAS strain rate recordings, we take two broadband stations in the array separated by a distance of 50 m and divide the difference between their respective waveform recordingsby their distance. Owing to the low noise floor of these shallow borehole seismometers, the resulting strain rate waveforms exhibit an extremely high SNR. [1]


---------

### Deep Learning Architecture
The deep learning architecture is based on the U-Net architecture.  We basically copied the architecture from the paper we mentioned before. You can find the code for the model on this [link](https://github.com/maniatisni/DAS-Denoising/blob/main/model.py#L142) as it is on this repository. Since PyTorch doesn't offer the option to use anti-aliasing CNNs, or anti-aliasing Max Pooling, we used the layers from [adobe's implementation](https://github.com/adobe/antialiased-cnns).

------------


### Training Strategy
The idea is to pretrain on the synthetic data till convergence, for 2000/3000 epochs, and then train on real world data, for fewer epochs, for example 50 and then use that trained model to infer on new events. As training loss we used Mean Squared Error, and to properly evaluate our results we used the local waveform coherence CC around the k-th DAS channel:

![](https://latex.codecogs.com/png.latex?%5Cmathrm%7BCC%7D_%7Bk%7D%3D%5Cfrac%7B1%7D%7B4%20N%5E%7B2%7D%7D%5Cleft%5B%5Csum_%7Bi%2C%20j%3D-N%7D%5E%7B&plus;N%7D%20%5Cmax%20%5Cleft%28%5Cfrac%7Bx_%7Bk&plus;i%7D%20*%20x_%7Bk&plus;j%7D%7D%7B%5Csqrt%7B%5Csum_%7Bt%7D%20x_%7Bk&plus;i%7D%5E%7B2%7D%20%5Csum_%7Bt%7D%20x_%7Bk&plus;j%7D%5E%7B2%7D%7D%7D%5Cright%29-2%20N-1%5Cright%5D)

The coherence gain is then defined as the local coherence computed for the J-invariant reconstruction divided by that of the input data. As such, coherence gains above 1 indicate that the reconstruction exhibits improved waveform coherence compared to the input data, which is beneficial for coherence-based seismological analyses template matching, beamforming).

