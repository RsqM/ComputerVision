from skimage import feature
import numpy as np
def DefectDetection(image):
    """
    Input: the Mask of the anomaly
    Output: the Defected Indices, and the edges asscociated with the anomaly
    Description:
    using the Mask, we apply a Canny Edge detection method which generates the edge map of the 
    anomalous region. This can later be used for Localisation and Bounding Box Depiction
    of the anomaly.
    rsqm
    """
    #5: Apply Canny edge detection to each image in the dataset
    defect_indices = []
    edges = feature.canny(image, sigma=5)
    edges2 = feature.canny(image, sigma=3)   # Adjust the sigma value as needed
    if edges.astype(np.uint8).mean().mean() < 00.0011:
      edges = edges2
    return defect_indices, edges