# Real Evaluation for Designing Sensor Fusion in UAV Platforms

This tool allow selecting the best fusion system parameters, for a given sensor configuration and a predefined real mission of UAV with a PX4 controller\*, which does not require ground truth.

\* Only for PX4 <= 5X

The **inputs** of this program are
are the logs of the flights performed for the designed mission. 

The **outputs** are:
* The modified parameters.
* The rms of the innovations in each axis of: position, velocity and magnetometer.
* Different figures that allow to compare and visualize these metrics on each flight based on different aspects.

![Picture1](https://user-images.githubusercontent.com/108266824/179523860-50f8d8d8-c421-47b0-8b0c-37a6b6bdacea.svg)


## Requirements
* Programs:
    * Python == 3.8.10
* Required libraries:
    * pandas     == 1.3.3
    * navpy      == 1.0
    * matplotlib == 3.5.0b1

\* for more information look requirements.txt file
