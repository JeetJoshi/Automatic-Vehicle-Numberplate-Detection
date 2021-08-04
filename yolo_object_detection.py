import cv2
import numpy as np
import glob
import random
import os
import datetime
import csv


for i in range (1,11):
    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
    classes = ["Number_plate"]
    images_path = glob.glob(f".\\cars\\c{i}.jpg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    random.shuffle(images_path)

    for img_path in images_path:

        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape


        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)


        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:

                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)


                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    numplt1 = img[y:y+h,x:x+w]
                    #cv2.imshow('plate1',numplt1)
        numplt = cv2.resize(numplt1,(300,50))
        numplt = cv2.blur(numplt,(1,1))
        numplt = cv2.cvtColor(numplt,cv2.COLOR_BGR2GRAY)
        numplt = np.array(numplt)
        size = numplt.shape[0]
        size1 = numplt.shape[1]
        for i in range(size):
            for j in range(size1):
                if numplt[i][j]<110: 
                    numplt[i][j] = 0
                else:
                    numplt[i][j] = 255
        #cv2.imshow('plate',numplt)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_ITALIC
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0,255,0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), font, 0.5, color, 2)
        cv2.imwrite('tr.jpg',numplt)
        os.system('tesseract .\\tr.jpg .\\output.txt') 
        img = cv2.resize(img,(500,300))

        f = open('output.txt.txt','r')
        number = f.read()
        print(number)
        dt = datetime.datetime.now()
        dt = dt.strftime("%d/%m/%Y %H:%M:%S")
        data = [dt,number]
        print(data)

        with open('data.csv', 'a') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)

        cv2.imshow("Image", img)
        key = cv2.waitKey(0)

        
        


    cv2.destroyAllWindows()