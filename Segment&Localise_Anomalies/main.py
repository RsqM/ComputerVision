import matplotlib.pyplot as plt
from src.ImageRead import ReadImage
from src.LearningAlgo import UsADusingGMM
from src.Binarize import OtsuThreshold
from src.Mask import Masking
from src.DefectDetect import DefectDetection
from src.Localisation import Localise

def Visualize(image, edges, regions):
# Step 6: Display the results
    fig = plt.figure(figsize = (15,15))
    ax0 = fig.add_subplot(2,4,1)
    ax0.imshow(image)
    ax0.set_title("Original(cvt. Gray)")
    ax1 = fig.add_subplot(2,4,2)
    ax1.set_title("Generated Mask")
    ax1.imshow(mask)
    ax2 = fig.add_subplot(2,4,3)
    ax2.set_title("Localization and Marking")
    ax2.imshow(image)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='yellow', linewidth=1.5, alpha = 0.6)
        ax2.add_patch(rect)
        
    ax3 = fig.add_subplot(2,4,4)
    ax3.set_title("Edges")
    ax3.imshow(image)
    ax3.imshow(edges, cmap = "Reds", alpha = 0.6)

#returns a grayscale and blurred image of the target image
gray, blurred = ReadImage()

#performs UsAD using GMM and returns an anomaly score for the image
anomaly_scores = UsADusingGMM(blurred)

#binarizes image and performs otsu's thresholding
anomalies = OtsuThreshold(anomaly_scores)

#generates a mask for the anomalous region
mask = Masking(blurred, anomalies, gray)

#detects nay defect if present, and creates an edge map
defect_indices, edges = DefectDetection(mask)

#generates coordinates for bounding box
regions = Localise(edges)

#generates a side-by-side plot of the processed image
Visualize(gray, edges, regions)