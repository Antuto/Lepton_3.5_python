import numpy as np
import cv2
from pylepton.Lepton3 import Lepton3
#from picamera2 import Picamera2

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

cv2.startWindowThread()

dim_x = 800
dim_y = 480
center = (int(dim_x/2),int(dim_y/2))
text_center = (int(dim_x/2)+20,int(dim_y/2)+20)
sensor_center = (80,60)

winname = "flir_windows"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
cv2.moveWindow(winname, 0, 0)
cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

"""# Configure camera and start
picam2 = Picamera2()
config = picam2.create_preview_configuration(raw={"size": (2592, 1944)}) #Max resolution : (3280, 2464)
picam2.configure(config)
picam2.start()
"""
def zoom_center(val,img):

    y_size = img.shape[0]
    x_size = img.shape[1]
    
    # define new boundaries
    x1 = int(0.5*x_size*(1-1/val))
    x2 = int(x_size-0.5*x_size*(1-1/val))
    y1 = int(0.5*y_size*(1-1/val))
    y2 = int(y_size-0.5*y_size*(1-1/val))

    # first crop image then scale
    img_cropped = img[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=val, fy=val)


def ktoc(val):
  return (val - 27315) / 100.0

while True:
    
    with Lepton3() as l:
        frame, _ = l.capture()
        """h, w, _ = frame.shape
        center_temp = frame[h//2, w//2]"""
        max_temp = np.max(frame)
        hot_y, hot_x, _ = np.unravel_index(np.argmax(frame), frame.shape)
        hot_x = (160-hot_x)*5
        hot_y = (120-hot_y)*4

        min_temp = np.min(frame)
        cold_y, cold_x, _ = np.unravel_index(np.argmin(frame), frame.shape)
        cold_x = (160 - cold_x) * 5
        cold_y = (120 - cold_y) * 4

        max_temp_text = f"Temp: {ktoc(max_temp)}"
        min_temp_text = f"Temp: {ktoc(min_temp)}"

    frame = cv2.normalize(frame, frame, 0, 60535, cv2.NORM_MINMAX)  # extend contrast
    frame = np.right_shift(frame, 8, frame)  # fit data into 8 bits

    # Récupération du pixel central
    line_size = 10



    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # Upscale the image using new  width and height

    up_width = 160 * 5
    up_height = 120 * 4
    up_points = (up_width, up_height)
    resized_up = cv2.resize(frame, up_points, interpolation=cv2.INTER_LANCZOS4)
    #Remove noise with blur then Sharpen it
    smoothed = cv2.GaussianBlur(resized_up, (3, 3), 10)
    image_sharp = cv2.filter2D(src=smoothed, ddepth=-1, kernel=kernel)
    
    final = np.uint8(image_sharp)
    #final = cv2.equalizeHist(final) #-> NEED TO TEST instead of normalize 
    rgb_img = cv2.applyColorMap(final, cv2.COLORMAP_PLASMA)

    cv2.line(rgb_img, (hot_x, hot_y - line_size), (hot_x, hot_y + line_size), (255, 255, 255), 1)
    cv2.line(rgb_img, (hot_x - line_size, hot_y), (hot_x + line_size, hot_y), (255, 255, 255), 1)
    cv2.circle(rgb_img, (hot_x, hot_y), 5, (255, 255, 255), 1)
    cv2.putText(rgb_img, max_temp_text, (hot_x+20, hot_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 255), 1)

    cv2.line(rgb_img, (cold_x, cold_y - line_size), (cold_x, cold_y + line_size), (255, 255, 255), 1)
    cv2.line(rgb_img, (cold_x - line_size, cold_y), (cold_x + line_size, cold_y), (255, 255, 255), 1)
    cv2.circle(rgb_img, (cold_x, cold_y), 5, (255, 255, 255), 1)
    cv2.putText(rgb_img, min_temp_text, (cold_x + 20, cold_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 128), 1)

    final_render = cv2.imshow(winname, rgb_img)
    if cv2.waitKey(1) == ord('q'):
        break
"""
    #----------------- STANDARD Camera-----------------------------
    #Capture raw input in an array
    normal = picam2.capture_array()
    zoomed = zoom_center(1.15,normal)
    #Resize image
    up_width1 = up_width
    up_height1 = up_height
    up_points1 = (up_width1, up_height1)
    resized_up1 = cv2.resize(zoomed, up_points1, interpolation=cv2.INTER_LANCZOS4)
    #convert raw BGR to RGB format
    rgb = cv2.cvtColor(resized_up1, cv2.COLOR_BGR2RGB)

    
    #----------Apply post process----------#
        # converting to gray scale
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        # remove noise
    kernel_size = 1
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    img_blur = cv2.GaussianBlur(gray, (1, 1), 0)
    
        # convolute with proper kernels
    #---------SOBEL---------#
    grad_x = cv2.Sobel(img_blur, ddepth, 1, 0, ksize=1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img_blur, ddepth, 0, 1, ksize=1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 1.0, abs_grad_y, 1.0, 0)
    #---------SOBEL - END---------#
    final_cam = cv2.cvtColor(grad,cv2.COLOR_GRAY2RGB)
    # shift the image 25 pixels to the right and 50 pixels down
    direction = np.float32([[1, 0, 60], [0, 1, -80]])
    shifted = cv2.warpAffine(final_cam, direction, (final_cam.shape[1], final_cam.shape[0]))

    #-----------------------------------
    #final_composite = cv2.addWeighted(rgb_img, 1, shifted, 1, 0.0)
    """

#cv2.VideoCapture(np.uint8(a))  # write it!



