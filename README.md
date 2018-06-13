# Obstacle Detection and Classification for Power Line Inspection Robots Based on Stereo Vision

<p align="justify">
The research aims to develop an obstacle recognition module to be used in a standalone robot for conducting visual inspection on transmission lines using a stereo camera as a data acquisition sensor. In this context, any element arranged in the transmission line that offers resistance to the passage of the robot is considered as an obstacle.</p>

<p align="justify">
The obstacles that will be taken into account during the development of the work and should be detected and classified by the module in question are suspension clamp, spacer damper and jumper.</p>

<p align="justify">
The process of recognizing obstacles will be divided into two stages. The first step will be responsible for detecting the obstacles located ahead of the robot, through image processing and conventional methods of extraction of characteristics. The second stage should perform the classification of the obstacle, detected during the previous stage, as one of the three possible obstacles, which will aid in the decision making of the robot and its autonomy throughout the inspection process. For the classification will be approached the convolutional neural network and support vector machine methods.</p>

<p align="justify">
The sensor that has been being used is the zed stereo camera that enables real-time high-resolution data acquisition and has some features, such as the depth map that reaches up to 20 meters, that might support future searches.</p>

<p align="justify">
The robotic framework used is Robot Operation System (ROS), Kinetic distribuition, along with the Ubuntu 16.04 version of the Linux operating system, which are executed in a processing and control unit - a computer that meets the minimal requirements operating.</p>

# Contents

  * [Requirements](#Requirements)
  * [Installation](#instala%C3%A7%C3%A3o)
  * [Usage ](#uso)
  * [Documentation](#documentação)
  * [Contributiton](#contribuicoes)
  * [Credits](#creditos)
  * [License](#licenciamento)
  * [Background IP](#background-ip)
  * [Acknowledgments](#acknowledgments)

# Requirements 

  * [Ubuntu 16.04 LTS (Xenial Xerus)](http://releases.ubuntu.com/16.04/) 
  * [Python 2.7](https://www.python.org/download/releases/2.7/) 
  * [ROS Kinetic](http://wiki.ros.org/kinetic)
