from scipy import linalg
import numpy as np
import cv2 as cv
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error as MSE

CUBE = cv.imread('cube.png')

def KFoldCrossVal(ground, OPENCVGround, Folds):
    kf = KFold(n_splits=Folds)
    
    mseScores = []
    
    for trainIndex, testIndex in kf.split(ground):
        groundTrain, groundTest = ground[trainIndex], ground[testIndex]
        OPENCVGroundTrain, OPENCVGroundTest = OPENCVGround[trainIndex], OPENCVGround[testIndex]
        
        mse = MSE(groundTest, OPENCVGroundTest)
        mseScores.append(mse)
        
    averageMse = np.mean(mseScores)
    
    return averageMse


def getGround(R, T, K, dist):
    
    # Creiamo la nostra griglia di punti sul pavimento (quindi z = 0)
    # Proiettiamo usando projectPoints da OpenCV e passiamo intrinseci ed estrinseci calcolati in precedenza
    
    xg = np.arange(-5, 10, 0.5) 
    yg = np.arange(-5, 10, 0.5) 
    xx, yy = np.meshgrid(xg, yg)

    dim = xx.shape[0]*xx.shape[1]
    points = np.zeros((dim,3), np.float32)

    xx = xx.reshape(dim)
    yy = yy.reshape(dim)

    points[:,0] = xx 
    points[:,1] = yy 
    points[:,2] = np.zeros((dim))
    
    ground, _ = cv.projectPoints(points, R, T, K, dist)
    ground = np.squeeze(ground).astype(np.int32)

    return ground

def drawGround(ground, windowName):
    img_to_show_res = CUBE.copy()
    for p in ground:
        img_to_show_res = cv.circle(img_to_show_res, p, 3, (255, 0, 0) )
    
    cv.imshow(windowName, img_to_show_res)
    while(True):
        k = cv.waitKey(500)
        if k == -1:  # if no key was pressed, -1 is returned
            continue
        else:
            break
    cv.destroyAllWindows()

def OPENCVCalibrateCamera(points):
    img = CUBE.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    objp, imgp = getPoints(points)
    
    # Setup calibrazione non planare di prova:
    # - Aggiungiamo una colonna vuota alla matrice usata per i punti dell'immagine, cioè z = 0
    # - Inizializziamo una matrice di camera K di prova
    # - Aggiungiamo i flag per una stima iniziale di intrinseche ed estrinseche 
    
    np.append(imgp, np.zeros((imgp.shape[0],1)), axis=1)
    
    K = initK(img.shape[:-1:])
    
    # L'utilizzo dei flag corretti si è rivelato fondamentale per una corretta proiezione dei punti
    
    return cv.calibrateCamera([objp], [imgp], gray.shape[::-1], K, None, flags=cv.CALIB_USE_INTRINSIC_GUESS|cv.CALIB_FIX_S1_S2_S3_S4| cv.CALIB_ZERO_TANGENT_DIST| cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3|cv.CALIB_USE_EXTRINSIC_GUESS)

# Definiamo i punti dell'oggetto usando coordinate 3D "false", usando un sistema che ha origine da un angolo del cubo, e i punti dell'immagine, misurati manualmente

def initK(shape):
    # initial guess of K
    K = np.zeros((3,3))

    K[0,0]=500
    K[1,1]=500
    K[2,2]=1
    K[0,2]=shape[0]/2
    K[1,2]=shape[1]/2
    return K

def getPoints(points):
    objp = np.zeros((points, 3), np.float32)
    imgp = np.zeros((points,2), np.float32)
        
    if points >= 6:
        
        # objp[0,:] sarà uguale all'origine del sistema di riferimento che piazziamo sul cubo
        objp[1,:] =[6, 0, 1]
        objp[2,:] =[7, 0, 7]
        objp[3,:] =[7, 7, 7]
        objp[4,:] =[3, 0, 4]
        objp[5,:] =[0, 7, 0]
        
        imgp[0,:] = [103, 254]
        imgp[1,:] = [253, 189]
        imgp[2,:] = [258, 39]
        imgp[3,:] = [155, 10]
        imgp[4,:] = [173, 132]
        imgp[5,:] = [18, 196]
        
        if points >= 12:
            
            objp[6,:] =[0, 7, 7]
            objp[7,:] =[0, 4, 4]
            objp[8,:] =[0, 0, 7]
            objp[9,:] =[4, 4, 7]
            objp[10,:] =[7, 0, 3]
            objp[11,:] =[0, 3, 7]
            
            
            imgp[6,:] = [14, 26]
            imgp[7,:] = [50, 120]
            imgp[8,:] = [103, 61]
            imgp[9,:] = [136, 29]
            imgp[10,:] = [254, 141]
            imgp[11,:] = [62, 46]
            
            if points == 24:
                
                objp[12,:] =[0, 7, 3]
                objp[13,:] =[7, 2, 7] 
                objp[14,:] =[1, 0, 1]   
                objp[15,:] =[2, 0, 1]   
                objp[16,:] =[3, 0, 1]   
                objp[17,:] =[4, 0, 1]   
                objp[18,:] =[5, 0, 1]   
                objp[19,:] =[6, 0, 1]   
                objp[20,:] =[1, 0, 2]   
                objp[21,:] =[2, 0, 2]   
                objp[22,:] =[3, 0, 2]   
                objp[23,:] =[4, 0, 2]   
                
                imgp[12,:] = [16, 127]
                imgp[13,:] = [224, 27]
                imgp[14,:] = [127, 221]
                imgp[15,:] = [149, 216]
                imgp[16,:] = [171, 210]
                imgp[17,:] = [193, 205]
                imgp[18,:] = [213, 199]
                imgp[19,:] = [234, 193]
                imgp[20,:] = [127, 194]
                imgp[21,:] = [149, 190]
                imgp[22,:] = [172, 184]
                imgp[23,:] = [193, 179]
    
    return objp, imgp

# Facciamo una stima della matrice di camera P a partire dalle corrispondenze tra punti dell'oggetto e dell'immagine

def estimateCameraMatrix(objp, imgp, points):
    A = np.zeros((points * 2, 12), dtype = np.float32)
    X, Y, Z = 0, 1, 2
    
    # Se un punto dell'immagine è uguale a un punto 3D per la matrice di proiezione che stiamo cercando P, allora possiamo trattare il sistema di equazioni riscritte come un problema di minimizzazione Ap = 0, per cui possiamo vincolare ||p||^2  = 1 e cercare un p tale che Ap si avvicini quanto più possibile a 0 e la norma di p si avvicini il più possibile a 1. Il fulcro della nostra ricerca è un problema agli autovalori, dal momento che vogliamo ricercare tramite SVD il valore singolare i-esimo più piccolo, in modo da, se la decomposizione ottenuta tramite SVD è uguale a M = USV^t, l'i-esimo vettore di V. Questa sarà la nostra matrice di proiezione 3x4.
    
    for i in range(points):
        A[2 * i:] = np.array([
                objp[i,X],
                objp[i,Y], 
                objp[i,Z], 
                1, 
                0, 0, 0, 0, 
                -imgp[i,X] * objp[i,X], 
                -imgp[i,X] * objp[i,Y], 
                -imgp[i,X] * objp[i,Z], 
                -imgp[i,X]
        ])
         
        A[2 * i + 1:] = np.array(
            [
                0, 0, 0, 0,
                objp[i,X],
                objp[i,Y], 
                objp[i,Z], 
                1,
                -imgp[i,Y] * objp[i,X], 
                -imgp[i,Y] * objp[i,Y], 
                -imgp[i,Y] * objp[i,Z], 
                -imgp[i,Y]  
            ]
        )
        
    _, _, Vt = np.linalg.svd(A)
    
    P = Vt[-1:]
    
    return P
    
def drawCubePoints(CUBE, imgp):
    for p in imgp:
        CUBE = cv.circle(CUBE, p.astype(np.int32), 3, (255, 0, 0))
    cv.imshow("CUBE", CUBE)
    cv.waitKey(0)
    cv.destroyAllWindows()    
    
def getKR(P):
    return  linalg.rq(P[:3,:3])

def getT(K, P):
    return np.matmul(np.linalg.inv(K), P[:,3])  