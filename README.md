# Gait Analysis
[_work in progress_] Accelerometer based gait analysis and identification.

---

This repository contains the very early stages of a system capable of recognizing a person based on gait analysis, using (at the time of writing) only accelerometer data. 

The data uploaded here are the first experiments, based on [HuGaDB](https://github.com/romanchereshnev/HuGaDB/). I've taken the data of the first ~20 steps of the first two people, using only the `y` component of the accelerometer on the right thigh, and built something that could discriminate between them. For this few training data, it seems to work well.

## Idea

In more detail, these are the main steps carried on for each person:
 1. Segment the accelerometer data isolating single steps. This is done using [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping) as distance function.
 2. Smooth extracted steps vectors with wavelet decomposition.
 3. Find a "baseline" step, which is a step representative of the person considered. For now it is the one that minimizes the sum of all distances between itself and the other steps, in future something like k-means can be used.


## Results

 - Two plots reporting various steps of two people respectively. The data on the y axis is the read of the `y` component of the accelerometer on the right thigh. The data is noisy, but you can tell that they were generated by different people.

<img src="https://github.com/MrSpadala/gait-analysis/blob/master/imgs/bunch_of_steps_01_smooth.png" width=600 />
<img src="https://github.com/MrSpadala/gait-analysis/blob/master/imgs/bunch_of_steps_03_smooth.png" width=600 />

 - Baseline step extracted for the two people

<img src="https://github.com/MrSpadala/gait-analysis/blob/master/imgs/baseline_step_01.png" width=600 />
<img src="https://github.com/MrSpadala/gait-analysis/blob/master/imgs/baseline_step_03.png" width=600 />

 - Below, we have taken test data of the same two people (not using during training), extract the steps, and calculate the distance between each step and the baseline steps. In the first plot, are plotted the distances between the baseline step of the first person and all the test steps. The blue points are the steps of the first person (and should have lower distance), the red ones are of the second person (and have higher distance). The same is done with the second plot, using the second person's baseline step, and this time the blue dots are the steps of the second person. 

<img src="https://github.com/MrSpadala/gait-analysis/blob/master/imgs/clf_0_smooth.png" width=600 />
<img src="https://github.com/MrSpadala/gait-analysis/blob/master/imgs/clf_1_smooth.png" width=600 />

We can see that the loss values are separated, and a simple classifier can be built.


## todo
 - improve step extraction
 - improve baseline selection
 - combine multiple measurements, both different accelerometers and gyroscopes
 - implement a simple classifier that can discriminate using the loss value (can be as simple as a threshold)
