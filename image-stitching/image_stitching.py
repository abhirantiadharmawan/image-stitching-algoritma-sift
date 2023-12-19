# Import Library
import cv2
import numpy as np
import sys

# Input Image
img_ = cv2.imread('00001.jpg') # Input the first image
img = cv2.imread('00002.jpg') # Input the second image

# Convert Grayscale Color
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY) # Convert the first image to grayscale
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convert the second image to grayscale

# SIFT Feature Detection
sift = cv2.SIFT_create() # Create a SIFT object
kp1, des1 = sift.detectAndCompute(img1,None) # Detect keypoints and compute descriptors for the first image
kp2, des2 = sift.detectAndCompute(img2,None) # Detect keypoints and compute descriptors for the second image

# Match keypoints using Brute Force Matcher
match = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # Create a Brute Force Matcher object
matches_bf = match.match(des1, des2) # Match descriptors of the two images

# Keep only the best 50 matches
num_matches = 50
img3 = cv2.drawMatches(img_, kp1, img, kp2, matches_bf[:num_matches], None) # Draw the best matches
good = sorted(matches_bf, key=lambda x: x.distance)[:num_matches] # Sort matches by distance and keep only the best 50

# Check matches matriks homografi
MIN_MATCH_COUNT = 49 # Minimum number of good matches required for homography estimation
if len(good) > MIN_MATCH_COUNT: # Check if there are enough good matches
    
    # Get keypoints coordinates of good matches
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # Estimate homography matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the first image using the homography matrix
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA) # Draw the warped image boundary on the second image
else:
    print("Not enough matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT)) # Print an error message if there are not enough matches

# Stitch two images
dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0])) # Warp the first image using the homography matrix
dst[0:img.shape[0],0:img.shape[1]] = img # Copy the second image into the stitched image

# Trim the stitched image
def trim(frame):
  
  # Check if the first row is entirely black
  if not np.sum(frame[0]):
    # If yes, call the trim function again with the first row removed
    return trim(frame[1:])

  # Check if the last row is entirely black
  if not np.sum(frame[-1]):
    # If yes, call the trim function again with the last row removed
    return trim(frame[:-2])

  # Check if the first column is entirely black
  if not np.sum(frame[:, 0]):
    # If yes, call the trim function again with the first column removed
    return trim(frame[:, 1:])

  # Check if the last column is entirely black
  if not np.sum(frame[:, -1]):
    # If yes, call the trim function again with the last column removed
    return trim(frame[:, :-2])

  # If no rows or columns are entirely black, return the image
  return frame

# Call the trim function to trim the stitched image
dst = trim(dst)

# Save stitched image
cv2.imwrite("hasilstitching99" + ".jpg", dst)
