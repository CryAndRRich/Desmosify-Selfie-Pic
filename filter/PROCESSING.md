# Desmosify Selfie Pic
The process of **image processing** and **converting** to black-and-white images, using **contrast enhancement techniques** and **edge detection algorithms** 

Since the **main purpose** of contrast enhancement techniques is to **improve image quality** to support the **output** of edge detection algorithms, we will start by discussing edge detection algorithms **first**

**Note:** The main purpose is for **learning**, everything is coded from **scratch**, **avoiding** the use of **built-in** functions, which results in **longer processing times** than usual

## Edge Detection Algorithms
An **edge detection algorithm** refers to a **technique** used in **image analysis** and **computer vision** to identify the locations of **significant edges** in an image while **filtering out false edges** caused by noise

In this project, **three edge detection algorithms** are used: **Canny** operator, **Marr-Hildreth** operator (LoG filter), and **Sobel** operator (The **process** of each algorithm can be **read and understood** through the codes and **comments** in each `.py` file)

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

**Another problem** is that, with certain images, edge detection algorithms **generate too many correct but unnecessary details**, the most typical example being **hair**. This makes [Step 2](https://github.com/CryAndRRich/Desmosify-Selfie-Pic/blob/main/README.md#step-2-desmosify) more **difficult** and **time-consuming**

![too_detailed](https://github.com/CryAndRRich/Desmosify-Selfie-Pic/blob/main/.github/too_detailed.png)

## Contrast Enhancement Techniques
**Image contrast enhancement** is a **pivotal process** in **digital image processing** that aims to **improve** the visibility and perceptibility of an image by **adjusting** and **amplifying** the difference in the **brightness** and **color** of the elements within the image

In this scenario, this manipulation **ensures** that the **distinct features** within an image are more easily **distinguishable**, helping the edge detection algorithm return **better results**

The file contains **three contrast enhancement techniques**: **IAGCWD** (Improved Adaptive Gamma Correction with Weight Distribution), **CLAHE** (Contrast Limited Adaptive Histogram Equalization), and **LA** (Luminance Adaptation) (You can read and **understand** each method **through the code and comments** in the `.py` files and the [references]())

Each technique has its own **specific utility**, however, this discussion will **focus solely** on the **impact** of these techniques on the **results** of edge detection algorithms, in this case, the **Canny operator**:

![Contrast_enhance_tech](https://github.com/CryAndRRich/Desmosify-Selfie-Pic/blob/main/.github/contrast_enhance_tech.jpg)

#### Some observes:
* **OG Canny** (apply canny to original image): Detects major edges **clearly** but struggles in **low-contrast areas**, missing finer details
* **IAGCWD**: **Significantly enhances brightness** and **highlights finer details**, especially in regions with **low contrast**. It **outperforms** OG Canny in detecting smaller and less **prominent edges**
* **CLAHE**: **Improves** edge detection in **complex areas** by enhancing local contrast. However, it may **introduce slight noise** in areas with **less information**
* **LA**: **Balances brightness effectively**, ensuring edges are **uniformly visible** across the image. While it maintains overall structure well, it **lacks the sharpness** provided by **IAGCWD** or **CLAHE**

**Overall conclusion**: **IAGCWD** is **effective** for **enhancing details** and detecting subtle edges, while **CLAHE** is **excellent** for **local contrast** improvement but slightly **prone to noise**, and LA is **well-balanced** but **less sharp** in edge detection

## References
Below are the **main articles and documents** on **contrast enhancement techniques**, which have **significantly contributed** to the development of the **code segments**

* Gamma Correction (IAGCWD):
  * [Contrast enhancement of brightness-distorted images by improved adaptive gamma correction](https://arxiv.org/pdf/1709.04427) 
* Histogram Equalization (CLAHE):
  * [Contrast Limited Adaptive Histogram Equalization](https://www.tamps.cinvestav.mx/~wgomez/material/AID/CLAHE.pdf)
* Luminance Adaption:
  * [Retinex-based perceptual contrast enhancement in images using luminance adaptation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8500743) 
