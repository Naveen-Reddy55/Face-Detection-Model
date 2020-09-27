import cv2

model=cv2.CascadeClassifier("\haarcascade_frontalface_default.xml\")
img1=cv2.imread("\input_image\")
resize=cv2.resize(img1,(600,600))
gray=cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
faces=model.detectMultiScale(gray,scaleFactor=1.05)
for x,y,w,h in faces:
    img=cv2.rectangle(resize,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow('Face_Detection',img)
cv2.waitKey(0)
cv2.destroyAllWindows()