from skimage.filters import threshold_otsu
def OtsuThreshold(anomaly_scores):
  """
  Input: Anomaly Scores, caluclated thorough Otsu's Thresholding Method
  Output: returns the anomalous region
  Description:
  Takes in the anomaly scores and binarizes the image to generate a greater contrast between the fg, and bg
  the anomalies are then calculated based on the threshold.
  rsqm
  """
  #3: Thresholding to identify anomalies
  threshold = threshold_otsu(anomaly_scores)
  anomalies = anomaly_scores > threshold
  return anomalies