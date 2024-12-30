# Desmosify Selfie Pic
A simple **selfie-to-Desmos graph** converter that runs locally, a process I personally call **"Desmosify"**

![result](https://github.com/CryAndRRich/Desmosify-Selfie-Pic/blob/main/.github/result.jpg)

# How the code works
## An overview of how everything works:
To convert an image into **usable data** for **plotting** on Desmos, we use a Python library called **Potrace**: it takes **PNG data** as a bitmap and returns a **traced path** of all the **Bézier curves**. Once we have our four points, we simply **plug them into the equation** and write it as a **LaTeX expression**, which is what the **Desmos API** uses to draw graphs. 

However, **another issue** arises due to the **edge detection algorithm** used by **Potrace**, the input image must **strictly** consist of **two** colors, such as black and white. This **necessitates** a **preprocessing step** to convert the image into a **black-and-white format**

Thus, there are **two main steps**: **image processing** and **converting the image into a Desmos graph**. They will be elaborated below

## Step 1: Image Processing
First, there is an **optional step**: taking a photo. If you **already** have an image, simply **rename** it and **place it** in the `samples` folder. Otherwise, the code will run, open the **computer's camera**, and **take a photo** for you. Once the selfie is **captured**, the image will **go through a processing stage** using **edge detection methods**

From the image below, there are **three contrast enhancement techniques** and **three edge detection algorithms**. Starting from the **original image**, you can **freely** choose **one or multiple methods**, click the `Apply Filters` button and achieve the result you find most **satisfactory**

![step1](https://github.com/CryAndRRich/Desmosify-Selfie-Pic/blob/main/.github/step1.png)

A more **detailed analysis** of the contrast enhancement techniques and the three edge detection algorithms can be found [here](https://github.com/CryAndRRich/Desmosify-Selfie-Pic/blob/main/filter/README.md)

## Step 2: Desmosify
Once you have a **suitable black-and-white image**, proceed by clicking the `Begin Rendering` button

Initially, there will be a line showing the **number of Bézier curves** and some notes (if the number is **too large**, **consider** changing the image or reducing the quantity as it may cause the device to **crash**). Start by **clicking** the `Start Rendering` button and wait (the **rendering time** will be **proportional** to the **number of Bézier curves**)

Once completed, you can click the "Capture Screenshot" button, and **an image will appear below**. Clicking on it will **automatically download** it to your device

![step2](https://github.com/CryAndRRich/Desmosify-Selfie-Pic/blob/main/.github/step2.png)



