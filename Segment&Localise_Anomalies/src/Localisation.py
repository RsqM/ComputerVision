from skimage.measure import label, regionprops

def Localise(edges):
    """
    Input: The Constructed Edge map using Canny Edge Detection
    Output: the Region Bounds of the defect for the bounding box
    Description:
    Takes the edge map and generates the coordinates for a boudning box.
    """
    #6 : Localisation of defect
    label_image = label(edges)
    regions = regionprops(label_image)
    return regions