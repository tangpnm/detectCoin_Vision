import cv2 
import numpy as np 
  
# Read image. 
img = cv2.imread('image/coins_01.jpg', cv2.IMREAD_COLOR)

d = 1024 / img.shape[1]
dim = (1024, int(img.shape[0] * d))
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

output = img.copy()
  
# Convert to grayscale. 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# improve contrast for differences lighting:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

def calcHistFromFile(file):
    img = cv2.imread(file)
    return calcHistogram(img)


# define Enum class
class Enum(tuple): __getattr__ = tuple.index

# separate the type to classifier in image
Material = Enum(('1Baht', '2Baht', '5Baht', '10Baht'))
  
# Blur image from gray picture using 7 * 15 kernel. 
gray_blurred = cv2.GaussianBlur(gray, (7,7), 0)
  
# Apply Hough transform on the blurred image. 
circles = cv2.HoughCircles(gray_blurred,  
                   cv2.HOUGH_GRADIENT, dp=2.2, minDist=30,
                           param1=200, param2=100, minRadius=50, maxRadius=120)

#cv2.imshow("Circle", gray_blurred)
count = 0

# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
                
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		
		count += 1
		
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.putText(output, "#{}".format(count), (int(x) - 8, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (150, 100, 126), -1)
 
# show the output image
cv2.putText(output, "Coins detected: {}".format(count),
            (5, output.shape[0] - 24), cv2.FONT_HERSHEY_PLAIN,
            3.0, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.imshow("output", np.hstack([img, output]))
cv2.waitKey(0)
