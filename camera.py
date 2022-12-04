import cv2
import torch
import time
import cv2
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def start():
    videoFrameCapture = cv2.VideoCapture(0)
    print(type(videoFrameCapture))
    videoFrameCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 224)#Establecemos el ancho
    videoFrameCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)#Establecemos el alto
    videoFrameCapture.set(cv2.CAP_PROP_FPS, 20)#Establecemos los frames/seg

    processImage = transforms.Compose([
        transforms.ToTensor(),  # Cada imagen la hacemos un tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalizamos la imagen para que el modelo la acepte
    ])

    model_trained = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model_trained.classifier=torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=1280, out_features=2),
    )
    model_trained.load_state_dict(torch.load("C:\\Users\\200an\Desktop\\pytorch-models\\models\\model-prueba-v0.0.2.pth"))
    model_trained.eval()
    print(type(model_trained))
    categories = {
        0: 'drowsiness',
        1: 'no-drowsiness'
    }
    model_trained.to("cpu")
    #Deshabilitamos el calculo de gradiente, para eficiencia de memoria
    with torch.no_grad():
        with torch.inference_mode():
            while True:
                now = time.time()
                isFrameOk, imageCapture = videoFrameCapture.read()

                #Si no se puede leer el frame, detenemos el programa
                if not isFrameOk:
                    raise RuntimeError("Error al leer el frame")

                imageCaptureGray = cv2.cvtColor(imageCapture, cv2.COLOR_BGR2GRAY)
                imageCaptureRGB = cv2.cvtColor(imageCaptureGray, cv2.COLOR_GRAY2RGB)
                imageCaptureRGB = cv2.resize(imageCaptureRGB, (224, 224))

                imageTensorTransformed = processImage(imageCaptureRGB)#Transformamos la imagen
                print(type(processImage))
                imageInputModel = imageTensorTransformed.unsqueeze(0)#Agregamos un mini-batch al principio

                modelOutput = model_trained(imageInputModel)

                #print(time.time() - now)
                probabilities = torch.nn.functional.softmax(modelOutput[0], dim=0)

                top_prob, top_id = torch.topk(probabilities, 2)

                for i in range(top_prob.size(0)):
                #cv2.putText(imageCaptureRGB,f"{categories[top_id[0].item()]}: {top_prob[0].item()*100:.2f}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                    print(f"{categories[top_id[i].item()]}: {top_prob[i].item()*100:.2f}")

                cv2.imshow("Predicciones", cv2.resize(imageCaptureRGB, (720, 480)))
                cv2.waitKey(1)

                print("\n")





