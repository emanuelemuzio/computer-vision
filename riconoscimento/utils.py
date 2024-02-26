import cv2 as cv
import glob, os
import pickle
import numpy as np
import math 

# Inizializzazione del classificatore per i volti usando l'xml per le  Haar-like Features dei volti dalla repository github di OpenCV

def buildClassifier(names):
    classifier_cascade = cv.CascadeClassifier()
    for name in names:    
        if not classifier_cascade.load(name):
            print(f"[ERROR] Errore durante l'inizializzazione del classificatore {name}")
    return classifier_cascade
        
# Caricamento dati già processati e salvati con pickle
        
def load(fname):
    f = open(fname,'rb')
    data = pickle.load(f)
    f.close()
    return data

def save(fname, data):
    f = open(fname,'wb')
    pickle.dump(data, f)
    f.close()
    return True

def normalize(matrix):
    normalizedData = (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))
    return normalizedData
        
# Recupero immagini per la costruzione dello spazio dei volti.
# Il dataset usato proviene dalle immagini già allineate di http://vis-www.cs.umass.edu/lfw/
# Da questo dataset sono state scelte randomicamente circa 4000 immagini, il numero minimo
# di immagini da considerare è di 2000

def getImgsDict(path, h_resize, w_resize, classifier):
    imgs = {}
    listdir = os.listdir(path)
    
    for dir in listdir:
        imgs[dir] = []
        for file in glob.glob(f"{path}/{dir}/*.jpg"):
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)
            faces = classifier.detectMultiScale(img, 1.1, 4)
            
            # Andiamo a fare il crop del volto nell'immagine SE è stata trovata una corrispondenza
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = img[y:y +h, x:x + w]
                
                # Resize dell'immagine e normalizzazione
                
                resized = cv.resize(face, (h_resize, w_resize), interpolation=cv.INTER_AREA)
                norm_resized = ((resized - np.min(resized)) / (np.max(resized) - np.min(resized)))
                                
                imgs[dir].append(norm_resized)
            
    return imgs
    
# Calcolo volto medio a partire da una lista di volti 64x64 elaborati con il metodo Viola/Jones
        
def getMeanFace(imgs):
    mean_face = np.zeros((64, 64), dtype=np.float32)

    for img in imgs:
        mean_face += img
        
    mean_face /= len(imgs)  
    
    return mean_face

# Metodo per l'annotazione del video raw

def createManualAnnotate(raw, manual, classifier):
    src = cv.VideoCapture(raw)
    output = None
    font = cv.FONT_HERSHEY_SIMPLEX
    
    # Lista dei volti noti
     
    labels = ['Francesco_Conti', 'Stefano_Corrao', 'Davide_Sgroi', 'Gabriele_Musso']
    menu = "\n 0) None \n 1) Francesco Conti \n 2) Stefano Corrao \n 3) Davide Sgroi \n 4) Gabriele Musso \n"
    
    fpsCap = 30
    currentFps = 0
    
    while src.isOpened() and currentFps < fpsCap:
        ret, frame = src.read()
        
        # Portiamo la nostra foto in bianco e nero per migliorare il rilevamento facciale
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        if output is None:
            height, width = frame.shape
            size = (width,height)
            output = cv.VideoWriter(manual, -1, 1, size)
                    
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        if currentFps < fpsCap:
            faces = classifier.detectMultiScale(frame)
            
            for (x, y, w, h) in faces:
                
                tmp = frame.copy()
                
                tmp = cv.rectangle(tmp, (x, y), (x + w, y + h), (0, 0, 0), 2)
                
                cv.imshow("Labeling", tmp)
                cv.waitKey(0)
                cv.destroyAllWindows()
                
                loop = True

                while loop:
                    try:
                        i = int(input(menu))
                        loop = i < 0 or i > 4
                    except ValueError:
                        print('\n Insert a valid option \n')
                
                if i > 0:
                    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    frame = cv.putText(frame, labels[i - 1], (x, y - 5),font, 1, (0, 0, 0), 2, cv.LINE_AA, False)
                    
            output.write(frame)
            currentFps += 1  
    src.release()
    output.release()
    
def detectFaces(raw, auto, classifier, knn, eigenfaces, m, known_ids, fpsCap, fps):
    src = cv.VideoCapture(raw)
    output = None
    font = cv.FONT_HERSHEY_SIMPLEX
    
    h_resize = 64
    w_resize = 64
    
    currentFps = 0
    
    frameList = []
    
    while src.isOpened() and currentFps < fpsCap:
        ret, frame = src.read()
        
        if frame is None:
            break
                
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        if output is None:
            height, width = frame.shape
            size = (width,height)
            output = cv.VideoWriter(auto, -1, fps, size)
                    
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        if currentFps < fpsCap:
            faces = classifier.detectMultiScale(frame)
            
            currentFrame = []
                                    
            for i in range(4):
                if i < len(faces):
                    face = faces[i]
                    (x, y, w, h) = face
                    faceROI = frame[y : y + h, x : x + w]
                    resized = cv.resize(faceROI, (h_resize, w_resize), interpolation=cv.INTER_AREA)
                    norm_face = ((resized - np.min(resized)) / (np.max(resized) - np.min(resized)))
                    
                    reshaped = np.reshape(norm_face, 4096)
                    
                    proj = eigenfaces @ (reshaped - m)
                    
                    # Vado a scorrere frame per frame per assegnare ai volti riconosciuti proiettati nel face space l'etichetta con prob. più alta
                    
                    label = knn.predict([proj])[0]
                    identity = known_ids[label]
                    identity_display = identity.split('_')[1]
                                    
                    currentFrame.append(identity)
                                                
                    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (250, 0, 0), 2)
                    frame = cv.putText(frame, f'{identity_display}', (x, y - 5),font, 1, (250, 0, 0), 2, cv.LINE_AA, False)
                else:
                    currentFrame.append("None")
                    
            output.write(frame)
            currentFps += 1  
            frameList.append(currentFrame)
    src.release()
    output.release()
    
    return frameList