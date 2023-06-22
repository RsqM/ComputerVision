import cv2
def ReadImage():
  '''
  Input: Path Of The Image.
  Output : A Grayscale, and a Blurred Instance of the image.
  Description: Imports a specific image from the path and converts it into grayscale
  Also applies a gaussian blur.
  rsqm
  '''
  #1: Read and preprocess the fabric image
  image = cv2.imread(f'{input("Input Path of Image: ")}')
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  return gray, blurred