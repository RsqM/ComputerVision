import numpy as np
def Masking(blurred, anomalies, gray):
  """
  Input: blurred image, gray image, and anomalies
  Output: Mask holding the anomaly
  Description: Generates a mask based on where the anomalies exist, 
  Masks out everything other than the anomaly.
  rsqm
  """
  #4: Segmentation and Localization
  # Create a binary mask of the detected anomalies
  mask = np.zeros_like(gray, dtype=np.uint8)
  mask[anomalies.reshape(blurred.shape)] = 255
  return mask