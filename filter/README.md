# Desmosify Selfie Pic
The process of **image processing** and **converting** to black-and-white images, using **contrast enhancement techniques** and **edge detection algorithms** 

Since the **main purpose** of contrast enhancement techniques is to **improve image quality** to support the **output** of edge detection algorithms, we will start by discussing edge detection algorithms **first**

**Note:** The main purpose is for **learning**, everything is coded from **scratch**, **avoiding** the use of **built-in** functions, which results in **longer processing times** than usual

## Edge Detection Algorithms
An **edge detection algorithm** refers to a **technique** used in **image analysis** and **computer vision** to identify the locations of **significant edges** in an image while **filtering out false edges** caused by noise

In this project, **three edge detection algorithms** are used: **Canny** operator, **Marr-Hildreth** operator (LoG filter), and **Sobel** operator (The **process** of each algorithm can be **read and understood** through the **comments** in each `.py` file)

Each method has its own **advantages** and **disadvantages** (though in terms of **efficiency**, the **Canny** operator is the most robust). The outputs of these three algorithms are then **combined** to produce **the most optimal result**

Moreover, since all algorithms are coded **from scratch** without using **built-in** functions, the results of the algorithms will **differ** from those obtained using **pre-built** functions. However, this difference is **not significant** and can sometimes even bring **certain benefits**

![Edge detection diff](https://github.com/CryAndRRich/Desmosify-Selfie-Pic/blob/main/.github/edge_detect_illustrate.jpg)

#### For example, from the table above:
* **The Canny detection** (from scratch) returns a result with more **clearly defined facial features**
* **The Marr-Hildreth** results are infact **the same**, the differences are simply due to a **snippet of code** in `marr_hildreth_edge_detection.py` that sets points above a **certain threshold** to 255 and reduces the rest to 0
* **The Sobel operator** (from scratch) produces a "weaker" result but **effectively reduces noise** in the area below the face

### Final result
The **final output** will be a **combination** of **one or more** of the methods mentioned above

**Currently**, the code uses the **majority voting approach**, which simply examines all pixels: a pixel is **considered an edge pixel** in the **final result** if the **majority** of methods (1/1, 2/2, 2/3) **classify** it as an **edge pixel**
### Current issues
There are still **some issues** at present:
* **Marr-Hildreth** and **Sobel** introduce **more noise** compared to Canny. As a result, when using **majority voting**, this noise **inadvertently gets included** in the final output
* **Sobel** produces results that **resemble a pencil sketch** rather than clear edges, which may initially appear to contain **more details** than Canny. However, the pixel values are **distributed across** a range from 0 to 255, making it **harder** to **distinguish edge pixels** and leading to **more noise** in the final result than **necessary**

One **potential solution** for achieving **better results** is to **incorporate weighting** for each method, **prioritizing** the results from **Canny**, while Marr-Hildreth and Sobel serve to **supplement small details** that Canny might **miss**. However, **determining** how to **assign the weights effectively** is a new challenge that requires further **testing** and **evaluation**


* Gamma Correction (IAGCWD):
  * [Contrast enhancement of brightness-distorted images by improved adaptive gamma correction](https://arxiv.org/pdf/1709.04427) 
* Histogram Equalization (CLAHE):
  * [Contrast Limited Adaptive Histogram Equalization](https://www.tamps.cinvestav.mx/~wgomez/material/AID/CLAHE.pdf)
* Luminance Adaption:
  * [Retinex-based perceptual contrast enhancement in images using luminance adaptation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8500743) 
