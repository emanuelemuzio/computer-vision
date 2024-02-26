import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Ricerca della strip 20x150 da utilizzare per l'analisi degli istogrammi dei valori di grigi
# Per la ricerca della strip faccio scorrere una finestra di dimensioni uguale alla strip lungo
# tutta l'immagine e vedo quale area ha la maggior concentrazione di pixel con intensità uguale al
# valore medio esatto di intensità

def getStrip(img):    
    (X, Y) = img.shape
    
    strip_dimX, strip_dimY = (20, 150)
    
    mean_intensity_value = np.around(np.mean(img))
    
    mean_count = 0
    posX, posY = (0, 0)
    true_point = (0, 0)
    
    while posX + strip_dimX < X and posY + strip_dimY < Y:
        area = img[posX: posX + strip_dimX, posY : posY + strip_dimY]
        pix_count = (area == mean_intensity_value).sum()
        if pix_count > mean_count:
            mean_count = pix_count 
            true_point = (posX, posY) 
        posX += strip_dimX
        posY += strip_dimY
        
    return img[true_point[0] : true_point[0] + strip_dimX, true_point[1]: true_point[1] + strip_dimY]

# Plot degli istogrammi di freq. delle intensità dei pixel
# Asse x: intensità del pixel - z
# Asse y: frequenza dell'intensità - pr(z)

def makeHistograms(data):
    imgs = list(data.keys())
    fig, axes = plt.subplots(len(imgs), 1, figsize=(7,14))
    fig.tight_layout()
    
    for i in range(len(imgs)):  
        img = imgs[i]
        counts, bins = np.histogram(data[img]['strip'])
        axes[i].set_title(img)
        axes[i].hist(bins[:-1], bins, weights=counts)

# The operation works like this: keep this kernel above a pixel, add all 
# the 25 pixels below this kernel, take the average, and replace the central 
# pixel with the new average value. This operation is continued for all the 
# pixels in the image. Try this code and check the result:   
 
def gaussianSmoothing(img, kernelSize):
    kernel = np.ones(kernelSize, np.float32)/25
    return cv.filter2D(img, -1, kernel)

# Per il restauro di immagini affette da salt-and-pepper noise scegliamo semplicemente 
# di fare median filtering sull'immagine e poi affinare i dettagli con un leggero sharpening

def impulseNoise(img):
    restored = cv.medianBlur(img, 3)
    sharpen = sharpening(restored)

    return sharpen

# Per il restauro delle immagini affette da rumore Gaussiano possiamo usare la 
# funzione built-in di opencv fastNlMeansDenoising andando ad agire sul parametro h
# che influisce sull'intensità del filtro: aumentandolo dal valore di default 7 a 10 riusciamo
# a ottenere un risultato più pulito sacrificando una quantità bassa di dettagli dell'immagine
# che risulta più levigata

def gaussianNoise(img):
    blur = cv.fastNlMeansDenoising(img, None, 10, 7, 21)

    return blur

# Per ovviare al problema del rumore uniforme applichiamo un avg. filter con kernel 5x5

def poissonNoise(img):
    blur = cv.bilateralFilter(img,9,75,75)
    
    return blur

def sharpening(img):
    
    sharpen_kernel = np.array([
        [0, -1, 0], 
        [-1, 5, -1], 
        [0, -1, 0]
    ])
    
    sharpen = cv.filter2D(img, -1, sharpen_kernel)
    
    return sharpen
    
def reduceNoise(noise_list, imgs):
    result = []

    for noise, img in zip(noise_list, imgs):
        restored_img = None
        if noise == 'gaussian':
            restored_img = gaussianNoise(img)
        elif noise == 'poisson':
            restored_img = poissonNoise(img)
        elif noise == 'impulsive':
            restored_img = impulseNoise(img)
        result.append(restored_img)
    return result

def deltaICM(a, b):
   if a == b:
       return 1
   else:
       return 0

# Data term

def computeEd(d, f):
    (W, H) = d.shape
    Ed = np.zeros((W, H))
        
    for i in range(1, W - 1):
        for j in range(1, H - 1):
            Ed[i][j] = -deltaICM(d[i][j], f[i][j])
    
    return Ed 

# penalty term

def computeEp(f):
    (W, H) = f.shape
    Ep = np.zeros((W, H))
        
    for i in range(1, W - 1):
        for j in range(1, H - 1):
            Ep[i][j] = -deltaICM(f[i][j], f[i - 1][j]) - deltaICM(f[i][j], f[i + 1][j]) - deltaICM(f[i][j], f[i][j - 1]) - deltaICM(f[i][j], f[i][j + 1]) 
    
    return Ep

def localEp(f):
    i, j = (1, 1)
    
    return -deltaICM(f[i][j], f[i - 1][j]) - deltaICM(f[i][j], f[i + 1][j]) - deltaICM(f[i][j], f[i][j - 1]) - deltaICM(f[i][j], f[i][j + 1])

def localEd(a, b):
    return -deltaICM(a, b)    

# Vedi libro paragrafo ICM

def ICM(d, MAX_ITER):
    count = 0
    d = cv.copyMakeBorder(d, 1, 1, 1, 1, cv.BORDER_REFLECT)
    (W, H) = d.shape
    f = np.copy(d)
    iter = 0
    changed = True
    while changed and iter < MAX_ITER:
        changed = False
        Ed = computeEd(d, f)
        Ep = computeEp(f)
        
        E = Ed + Ep
        
        # EVITARE CICLI FOR
        # Piuttosto che ragionare sui pixel e vedere le intensità
        # per ogni intensità guardo cosa succede nella mia matrice, e lavoro con numpy
        
        for i in range(1, W - 1):
            for j in range(1, H - 1):
                energy = 100
                valid = f[i][j]
                proposal = np.copy(f)
                            
                for intensity in range(0, 256, 32):
                    proposal[i][j] = intensity    
                    t = intensity
                    
                    local_f = proposal[i - 1 : i + 2, j - 1 : j + 2]
                    
                    Ed_ = localEd(d[i, j], proposal[i][j])
                    Ep_ = localEp(local_f)
                    E_ = Ed_ + Ep_
                    
                    if E_ < E[i][j] and E_ < energy:
                        energy = E_
                        valid = t
                        changed = True
                        count += 1
                f[i][j] = valid
        iter += 1
                
    return f
                
# img = cv.imread('denoising/gr1/Noisy/R63.bmp', cv.IMREAD_GRAYSCALE)
# low = impulseNoise(img)
# icm_test, pixel_changed = ICM(low, 3)
# cv.imshow(f'Pixel change: {pixel_changed}', icm_test)
# cv.waitKey(0)
# cv.destroyAllWindows()