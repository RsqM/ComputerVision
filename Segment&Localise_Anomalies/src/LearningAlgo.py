from sklearn.mixture import GaussianMixture
def UsADusingGMM(blurred):
  """
  Input: Blurred image
  Output: The anomaly scores associated to the image.
  Description:
  Performs Gaussian Mixture analysis on the image. This model is then used to score the sample
  on the blurred image which gives the anomaly scores.
  rsqm
  """
  #2: Unsupervised anomaly detection using Gaussian Mixture Models (GMM)
  gmm = GaussianMixture(n_components=2, covariance_type = "full")  # Assuming two classes: normal and defective
  gmm.fit(blurred.reshape(-1, 1))
  anomaly_scores = -gmm.score_samples(blurred.reshape(-1, 1))
  return anomaly_scores