import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

-------------------------------------------------------------------------------------------
def epi_line_draw(pic_1,pic_2,lines,pts1,pts2):
    image_1= cv2.cvtColor(pic_1,cv2.COLOR_BGR2GRAY)
    r,c = image_1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        pic_1 = cv2.line(pic_1, (x0,y0), (x1,y1), color,1)
        pic_1 = cv2.circle(pic_1,(int(pt1[0]),int(pt1[1])),5,color,-1)
        pic_2 = cv2.circle(pic_2,(int(pt2[0]),int(pt2[1])),5,color,-1)
    return pic_1,pic_2
#------------------------------------------------------------------------------------------
def compute_fund_matrix(rand,x,pad_1,pad_2):
    A = np.zeros((8,9))
    for idx in range(8):
        x_1 = pad_1[rand[x]][0]
        y_1 = pad_1[rand[x]][1]
        x_2 = pad_2[rand[x]][0]
        y_2 = pad_2[rand[x]][1]
        A[idx] = [x_1*x_2, x_1*y_2, x_1, y_1*x_2, y_1*y_2, y_1, x_2, y_2, 1]
        x += 1
    u,s,v_t = np.linalg.svd(A)
    fund = v_t[-1]
    fund = fund.reshape(3,3)
    u2,s2,v2_t = np.linalg.svd(fund)
    s[2] = 0
    fund = u2 @ np.diag(s2) @ v2_t
    fund = fund / fund[-1,-1]
    return fund
#-------------------------------------------------------------------------------------------------------
def Ransac_fn(pad_1, pad_2, threshold, inliers):
    cor_p = 0.9
    incorr_p = 0.7
    nps = 8
    x=0
    N = int(np.log(1-cor_p)/np.log(1-(1-incorr_p)**nps))
    np.random.seed(0)
    i=0
    s=np.random.choice(len(pad_1),N*8)
    inlier_obt = 0
    while i < N:
        F = compute_fund_matrix(s,x,pad_1,pad_2)
        err = pad_2 @ F @ pad_1.T
        err = abs(np.diag(err))
        for err in err:
            if err < threshold:
                inlier_obt+=1
        if inlier_obt > inliers:
            return F
        i += 1
        x += 8
#------------------------------------------------------------------------------------------------------

def image_processing(pic1, pic2):

    feature_detector = cv2.SIFT_create()
    val_1, param_1 = feature_detector.detectAndCompute(pic1,None)
    val_2, param_2 = feature_detector.detectAndCompute(pic2,None)
    matching_mechanism = cv2.BFMatcher()
    number_of_matches = matching_mechanism.knnMatch(param_1,param_2, k=2)

    good_set_1 = []
    for m,n in number_of_matches:
        if m.distance < 0.5*n.distance:
            good_set_1.append(m)
    
    points= []
    for match in good_set_1:
        x1, y1 = val_1[match.queryIdx].pt
        x2, y2 = val_2[match.trainIdx].pt
        points.append([((x1), (y1)), ((x2), (y2))])

    result = cv2.drawMatchesKnn(pic1,val_1,pic2,val_2,number_of_matches[:50],None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result, points

pic_1 = cv2.imread('im0c.png')
pic_2 = cv2.imread('im1c.png')

picture_1 = pic_1.copy()
picture_2 = pic_2.copy()

feature_img, points = image_processing(pic_1, pic_2)
pts1, pts2 = zip(*points)
cv2.imshow("result", feature_img)

cv2.waitKey(0)

ones =  np.ones((len(pts1),1))
pts1_padded = np.hstack([pts1,ones])
pts2_padded = np.hstack([pts2,ones])

F = Ransac_fn(pts1_padded, pts2_padded, threshold= 0.75, inliers=250)
print("Fundamental Matrix",F)
K_ar = np.array([[1733.74, 0, 792.27], [0, 1733.74, 541.89], [0, 0, 1]])

def triangulatepoints(R, t, K, points1, points2):
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K, np.hstack((R, t.reshape(-1, 1))))
    X = cv2.triangulatePoints(P1, P2, np.asarray(
        points1[0:4]).T, np.asarray(points2[0:4]).T)
    X = np.float32(X)
    X /= (X[3])
    k = 0
    for i in range(4):
        if X[2][i] >= 0:
            k += 1
    return k


def findRT(e, K, points1, points2):
    U, S, V = np.linalg.svd(e)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U@W@(V)
    R2 = U@(W.T)@(V)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    t1 = U[:, 2]
    t2 = -U[:, 2]
    k1 = triangulatepoints(R1, t1, K, points1, points2)
    k2 = triangulatepoints(R1, t2, K, points1, points2)
    k3 = triangulatepoints(R2, t1, K, points1, points2)
    k4 = triangulatepoints(R2, t2, K, points1, points2)
    maxk = max(k1, k2, k3, k4)
    if k1 == maxk:
        return R1, t1
    elif k2 == maxk:
        return R1, t2
    elif k3 == maxk:
        return R2, t1
    else:
        return R2, t2

E = K_ar.T @ F @ K_ar
# print("E matrix", E)
# r, T = findRT(E, K_ar, pts1, pts2)
# print("Rotation", r)
# print("Translation", T)

l1 = cv2.computeCorrespondEpilines(np.asarray(pts2).reshape(-1,1,2), 2,F)
l1 = l1.reshape(-1,3)
image_5,image_6 = epi_line_draw(picture_1,picture_2,l1,pts2,pts1)
l2 = cv2.computeCorrespondEpilines(np.asarray(pts1).reshape(-1,1,2), 1,F)
l2 = l2.reshape(-1,3)
image_3,image_4 = epi_line_draw(picture_2,picture_1,l2,pts2,pts1)
ret,H1,H2 = cv2.stereoRectifyUncalibrated(np.int32(pts1),np.int32(pts2),F,(pic_1.shape[1],pic_1.shape[0]))
print("H1:", H1)
print("H2:", H2)

corrected_picture_1 = cv2.warpPerspective(pic_1, H1, (pic_1.shape[1],pic_1.shape[0]))
corrected_picture_2 = cv2.warpPerspective(pic_2, H2, (pic_2.shape[1],pic_2.shape[0])) 

corrected_epipicture_1 = cv2.warpPerspective(picture_1, H1, (picture_1.shape[1],picture_1.shape[0]))
corrected_epipicture_2 = cv2.warpPerspective(picture_2, H2, (picture_2.shape[1],picture_2.shape[0])) 

gray_corrected_picture1 = cv2.cvtColor(corrected_picture_1,cv2.COLOR_BGR2GRAY)
gray_corrected_picture2 = cv2.cvtColor(corrected_picture_2,cv2.COLOR_BGR2GRAY)
H1, W1 = gray_corrected_picture1.shape


def squared_distances(i, j, l):
    dist = 0
    for a in range(-1, 2):
        for b in range(-1, 2):
            dist += (int(gray_corrected_picture1[i+a][j+b]) -
                         int(gray_corrected_picture2[i+a][b+j-l]))**2
    return dist

def climax(w1,h1, vmi, vmx, f, b):
    sq_dist = np.zeros(shape=(w1, h1))
    depth = np.zeros(shape=(w1, h1))
    for i in np.arange(3, H1-3):
        for j in np.arange(3, W1-3):
            SD_min = math.inf
            for l in np.arange(0, 20, 3):
                dist = squared_distances(i, j, l)
                SD_min = min(SD_min, dist)
                if dist == SD_min:
                    disp = l
            sq_dist[j][i] = disp
            if int(disp) == 0:
                disp = 0.0000000001
                depth[j][i] = f*b/disp
            else:
                depth[j][i] = f*b/disp
            


    disparity_norm = cv2.normalize(
        sq_dist.T, None, vmi, vmx, cv2.NORM_MINMAX, cv2.CV_8U)
    disp_colormap = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
    cv2.imshow("disparity_grayscale", disparity_norm)
    cv2.imshow("color_map_disp", disp_colormap)
    depth_normalized = cv2.normalize(
        depth.T, None, vmi, vmx, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    cv2.imshow("depth_grayscale", depth_normalized)
    cv2.imshow("color_map", depth_colormap)
    return None

cv2.imshow('picture_1',corrected_picture_1)
cv2.waitKey(0)
cv2.imshow('picture_2',corrected_picture_2)
cv2.waitKey(0)
cv2.imshow('picture_1_epi',corrected_epipicture_1)
cv2.waitKey(0)
cv2.imshow('picture_2_epi',corrected_epipicture_2)
cv2.waitKey(0)
art_room = climax(1920, 1080, 55, 142, 1733.74, 536.62)




cv2.waitKey() & 0xFF == ord("q")
cv2.destroyAllWindows()


