# Import Library
import cv2
import numpy as np
import sys

# Input Image
img_ = cv2.imread('00001.jpg') # Membaca gambar pertama
img = cv2.imread('00002.jpg') # Membaca gambar kedua

# Convert Grayscale Color
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY) # Mengubah gambar pertama menjadi skala abu-abu
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Mengubah gambar kedua menjadi skala abu-abu

# Mendeteksi Fitur SIFT
sift = cv2.SIFT_create() # Membuat objek SIFT
kp1, des1 = sift.detectAndCompute(img1, None) # Mendeteksi titik kunci dan menghitung deskriptor untuk gambar pertama
kp2, des2 = sift.detectAndCompute(img2, None) # Mendeteksi titik kunci dan menghitung deskriptor untuk gambar kedua

# Pencocokan Keypoints Menggunakan Brute Force Matcher
match = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # Membuat objek Brute Force Matcher
matches_bf = match.match(des1, des2) # Mencocokkan deskriptor dari dua gambar

# Pencocokan gambar Terbaik
num_matches = 50
img3 = cv2.drawMatches(img_, kp1, img, kp2, matches_bf[:num_matches], None) # Menggambar pencocokan terbaik
good = sorted(matches_bf, key=lambda x: x.distance)[:num_matches] # Mengurutkan pencocokan berdasarkan jarak dan menyimpan hanya 50 yang terbaik

# Pencocokan Matriks Homografi
MIN_MATCH_COUNT = 49 # Jumlah pencocokan yang baik yang diperlukan untuk estimasi homografi
if len(good) > MIN_MATCH_COUNT: # Memeriksa apakah terdapat cukup pencocokan yang baik
    
    # Dapatkan koordinat titik kunci dari pencocokan yang baik
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Estimasi matriks homografi
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Memutar gambar pertama menggunakan matriks homografi
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) # Menggambar batas gambar yang diputar pada gambar kedua
else:
    print("Tidak cukup pencocokan yang ditemukan - %d/%d", (len(good) / MIN_MATCH_COUNT)) # Menampilkan pesan kesalahan jika tidak cukup pencocokan

# Penggabungan Gambar
dst = cv2.warpPerspective(img_, M, (img.shape[1] + img_.shape[1], img.shape[0])) # Memutar gambar pertama menggunakan matriks homografi
dst[0:img.shape[0], 0:img.shape[1]] = img # Menyalin gambar kedua ke dalam gambar yang disatukan

# Pemangkasan Gambar
def trim(frame):
  
  # Periksa apakah baris pertama seluruhnya hitam
  if not np.sum(frame[0]):
    # Jika ya, panggil fungsi trim lagi dengan membuang baris pertama
    return trim(frame[1:])

  # Periksa apakah baris terakhir seluruhnya hitam
  if not np.sum(frame[-1]):
    # Jika ya, panggil fungsi trim lagi dengan membuang baris terakhir
    return trim(frame[:-2])

  # Periksa apakah kolom pertama seluruhnya hitam
  if not np.sum(frame[:, 0]):
    # Jika ya, panggil fungsi trim lagi dengan membuang kolom pertama
    return trim(frame[:, 1:])

  # Periksa apakah kolom terakhir seluruhnya hitam
  if not np.sum(frame[:, -1]):
    # Jika ya, panggil fungsi trim lagi dengan membuang kolom terakhir
    return trim(frame[:, :-2])

  # Jika tidak ada baris atau kolom yang seluruhnya hitam, kembalikan gambar
  return frame

# Panggil fungsi trim untuk memotong gambar yang disatukan
dst = trim(dst)

# Menyimpan Gambar Hasil
cv2.imwrite("hasilstitching" + ".jpg", dst)
