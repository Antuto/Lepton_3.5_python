import cv2
import numpy as np

dim_x = 800
dim_y = 408
center = (int(dim_x/2),int(dim_y/2))

line_size = 10

print(center)

# Create a black image
img = np.zeros((dim_y, dim_x, 3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
cv2.line(img, (center[0], center[1]-line_size), (center[0], center[1]+line_size), (255, 255, 255), 1)
cv2.line(img, (center[0]-line_size, center[1]), (center[0]+line_size, center[1]), (255, 255, 255), 1)
cv2.circle(img,center, 5, (255,255,255), 1)

winname = "flir_windows"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
cv2.moveWindow(winname, dim_x - 1, dim_y - 1)
cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

cv2.imshow(winname,img)
cv2.waitKey(0)
cv2.destroyAllWindows()