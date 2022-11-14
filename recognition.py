import cv2
import numpy as np
import os


def lettura(path, size):
    nomi = []
    #immagini di training
    training_x = []
    #etichette di training
    training_y = []
    etichetta = 0
    #generazione dei nomi dei file appartenenti alla cartella iniziale
    for root, directories, files in os.walk(path):#esplorazione albero
        for dir in directories:
            nomi.append(dir)
            sub_path = os.path.join(root, dir)
            #esamino le immagini delle sottocartelle
            for file in os.listdir(sub_path):
                img = cv2.imread(os.path.join(sub_path, file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                #ridimensionamento dell'immagine
                img = cv2.resize(img, size)
                #generazione dei dati di training
                training_x.append(img)
                training_y.append(etichetta)
            #le etichette sono dei numeri
            etichetta += 1#ogni nome ha una etichetta associata monica 0, rachel 1

    training_x = np.asarray(training_x, np.uint8)#conversione in array np
    training_y = np.asarray(training_y, np.int32)
    return nomi, training_x, training_y
            
    

if __name__ == "__main__":
    path = '/Users/ilariagallo/Desktop/ApplicazioneRec/Dataset'
    size = (200, 200)
    nomi, training_x, training_y = lettura(path, size)


    model = cv2.face.EigenFaceRecognizer_create()
    model.train(training_x, training_y)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


    #test_img = '/Users/ilariagallo/Desktop/ApplicazioneRec/Bill_Gates.jpeg'
    test_img = '/Users/ilariagallo/Desktop/ApplicazioneRec/Bill_Clinton_0004.jpg'
    img = cv2.imread(test_img)
    #il rilevamento del volto avviene sull'immagine in scala di girigi
    #in seguito gli elementi rilevati sono evidenzianti tramite cornici sull'immagine originale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    volti = face_cascade.detectMultiScale(img, scaleFactor=1.3,minNeighbors= 5)
    
    #viene creata la cornice relativa all'immagine rilevata
    for(x, y, widht, height) in volti:
        cv2.rectangle(img,(x,y),(x+widht,y+height),(0,255,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        regione_gray = img_gray[y:y + height, x:x + widht]
        if regione_gray.size == 0:
        #significa che è  vuota
            continue
            
        regione_gray = cv2.resize(regione_gray, size)
        etichetta, confidence = model.predict(regione_gray)
        
        nomeIdentificato = nomi[etichetta]
        cv2.putText(img,nomeIdentificato,(x,y),font,0.8,(0,0,0),2)
        
        cv2.imshow('Face Recognition', img)
        cv2.waitKey(10000)


