# **Finding Lane Lines on the Road** 

## Goals
**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.
1. First, I converted the images to grayscale, applied gussian blurring to denoise the image and then used canny fuction to detect edges fromt the images
2. Extract the region of interests using Polygons, the four verteces for defing the polygon are following: 
```python

LR_marginal = 5 * imshape[1] / 12
UD_marginal = 7 * imshape[0] / 11
# vertices order. Bottom Left -> Up Left -> Up Right -> Bottom Right
vertices = np.array([[(50, imshape[0]),
                      (LR_marginal, UD_marginal),
                      (imshape[1]-LR_marginal, UD_marginal),
                      (imshape[1]-50, imshape[0])]], dtype=np.int32)
```
3. Use HoughLinesP function to detect edges and lines.
4. Draw a single line on the left and right lanes. However, in order to draw a single line on the left and right lanes, I modified the draw_lines() function to have several modules:
  * First, seperate lines from left and right lanes. Here three criteria are considered: a) lines with slopes whose abosolute value are less than 0.5 dropped; 2) slope < 0, then the line belongs to left lanes, slope > 0, then the line belongs to right lanes; 3) for a potential false positive, such a line with negative slope, if both the two dots defining this line are closer to the right boundry of the image, then it is dropped.
  * Second, once we seperated the lines, we fit a line for each group using cv2.fitLine functions. 
  * Third, find the maximum points (end points) inside the region of interests for each line and extrapolate over the maximum points
5. For the optinal challenge, from the video we can know it has some frames that hardly find lanes. The ituition to solve this is to smooth the line from historical informations. In my work, I stored a historical line (previous line) for left and right lanes, if no line found in current frame, or we notice that line slope change very rapid in the frame, we adopt historical line, otherwise we set line value as a combination of current line and historical line. some parameters are:

```python

""""
Either of following cases occurs, current information will be dropped, history
will adopted directly
1. Current slope is not accessible that might be caused by none points found
   by HoughLinesP algorithm
2. Current slope changes too much compared with history

VARIABLE:
   HISTORY: a tuple (slope, b) defined a history/previous line
   USE_HISTORY: whether using smooth
   ALPHA: combination ratio of current line and previous line
   MAX_ANGLE_CHANGE: the maximum angle change

NOTE: please initialize HISTORY before use.
"""
HISTORY = [[], []]
USE_SMOOTHING = False
ALPHA = 0.8
MAX_ANGLE_CHANGE = math.pi / 3

```

**More details kind be find at the [notebook file](P1.ipynb) or [html file](P1.html)**


### 2. Identify potential shortcomings with your current pipeline

1. The parameter setting for edge detection seems not that robust, especially for bad whether
2. The whole pipeline might be vulnerable to sharp turns. 


### 3. Suggest possible improvements to your pipeline

1. For bad whether, we might need to manipulate the image to make the edge more clear and easier to detect
2. Region of interests should adjusted according to cars position or camera should be adjusted
3. Smoothing should take more information, such as slope change speed, the change from both right or left lane might help the other.
