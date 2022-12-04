import torch
from torchvision.models import mobilenet_v2,MobileNet_V2_Weights
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import cv2
import PIL

class ModelMobileNetv2(torch.nn.Module):

    def __init__(self, trainingFolderPath="", testFolderPath="",batch_size=2):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"#Verificamos si el equipo cuenta con CUDA
        self.trainning_folder = trainingFolderPath
        self.test_folder = testFolderPath
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

        #Congelamos los parametros
        for param in self.model.parameters():
            param.required_grad = False

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         weight_decay=0.1,
                                         lr=0.001,
                                         momentum=0.9)#Configuramos el optimizador

        self.model.optimizer = self.optimizer#Añadimos el optimizador al modelo
        self.loss_function = torch.nn.CrossEntropyLoss()#Definimos la funcion de perdida
        self.batch_size = batch_size
        self.training_dataset = datasets.ImageFolder(root=self.trainning_folderolder,#Obtenemos el dataset
                                                     transform=self.imageTransform(),
                                                     target_transform=None)
        self.test_dataset = datasets.ImageFolder(root=self.test_folder,
                                                 transform=self.imageTransform())#Obtenemos el dataset de test
        self.training_dataloader = DataLoader(dataset=self.training_dataset, #Creamos el DataLoader
                                              batch_size=self.batch_size,
                                              num_workers=2,
                                              shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                          batch_size=1,
                                          num_workers=1,
                                          shuffle=False)#Creamos el DataLoader de el test dataset
        self.classes = self.training_dataset.classes
        print(f"Clases del modelo => {self.classes}")
        self.classes_dict = self.training_dataset.class_to_idx

        #Cambiamos el tamaño de respuestas de salida a output_model, y las demas caracteristicas las dejamos igual
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=1280, out_features=len(self.classes)),
        )

        self.categories = {
                            0: 'drowsiness',
                            1: 'no-drowsiness'
                          }
    def imageTransform(self):
        return transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(), # Cada imagen la hacemos un tensor
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# Normalizamos la imagen para que el modelo la acepte
                ])

    def training_step(self):
        self.model.train()#Establecemos el modelo en modo entrenamiento

        #Creamos las variables para la perdida de entrenamiento y la eficacia del entrenamiento
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(self.training_dataloader):
            #Enviamos los datos al CPU
            X, y = X.to(self.device), y.to(self.device)

            # 1. Forward pass
            y_pred = self.model(X)

            #Calculamos la funcion de perdida y lo acumulamos
            loss = self.loss_function(y_pred, y)
            train_loss += loss.item()

            #Establecemos el gradiente del optimizador en 0
            self.optimizer.zero_grad()

            #Computarizamos los gradientes
            loss.backward()

            #Actualizamos los parametros
            self.optimizer.step()

            # Calculamos y acumulamos la eficiencia en todos los batches
            # Con torch.argmax obtenemos el tensor de mayor valor
            # Con torch.softmax obtenemos las probabilidades [0,1] y su suma debe de ser 1
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        #Obtenemos el promedio de la pérdida y la eficiencia del entrenamiento
        train_loss = train_loss / len(self.training_dataloader)
        train_acc = train_acc / len(self.training_dataloader)

        return train_loss, train_acc

    def testing_step(self):
        self.model.eval()  # Establecemos el modelo en modo evaluacion
        #Creamos las variables para la perdida de entrenamiento y la eficacia del entrenamiento
        testing_loss, testing_acc = 0, 0

        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.test_dataloader):
                # Enviamos los datos al CPU
                X, y = X.to(self.device), y.to(self.device)

                test_y_pred = self.model(X)

                # Calculamos la funcion de perdida y lo acumulamos
                loss = self.loss_function(test_y_pred, y)
                testing_loss += loss.item()

                predicted_labels = test_y_pred.argmax(dim=1)
                testing_acc += ((predicted_labels == y).sum().item()/len(predicted_labels))

        # Obtenemos el promedio de la pérdida y la eficiencia del entrenamiento
        testing_loss = testing_loss / len(self.test_dataloader)
        testing_acc = testing_acc / len(self.test_dataloader)

        return testing_loss, testing_acc
    def trainModel(self, epochs = 1):
        print("--------------ENTRENAMIENTO INICIALIZADO--------------")

        results = {
            "training_loss": [],
            "training_accuracy": [],
            "testing_loss" : [],
            "testing_accuracy" : []
        }

        for epoch in range(0, epochs):
            print(f"Epoch #{epoch+1}")

            train_loss, train_acc = self.training_step()
            test_loss, test_acc = self.testing_step()

            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"tests_acc: {test_acc:.4f} | "
            )

            results['training_loss'].append(train_loss)
            results['training_accuracy'].append(train_acc)
            results['testing_loss'].append(test_loss)
            results['testing_accuracy'].append(test_acc)

        torch.save(self.model.state_dict(), "C:\\Users\\200an\Desktop\\pytorch-models\\models\\model-prueba-v0.0.4.pth")

        print("--------------ENTRENAMIENTO FINALIZADO--------------")
        return results
    def getTrainningDataset(self):
        return self.training_dataset

    def testModel(self, pathFolderImages):

        self.model.load_state_dict(torch.load("C:\\Users\\200an\Desktop\\pytorch-models\\models\\model-prueba-v0.0.2.pth"))
        self.model.eval()

        correctPredictions = 0;
        totalPredictions = 0;
        for files in os.listdir(pathFolderImages):
            #Cada ciclo es una carpeta/clase diferente
            pathOfClass = f"{pathFolderImages}\\{files}"
            for frame in os.listdir(pathOfClass):
                classOfFrame = 0
                if files == "no-drowsiness":
                    classOfFrame = 1

                frameImage = cv2.imread(f"{pathOfClass}\\{frame}")
                processImage = self.imageTransform()
                frameImage = PIL.Image.fromarray(frameImage, "RGB")
                tensorImage = processImage(frameImage)

                inputModel = tensorImage.unsqueeze(0)

                output = self.model(inputModel)

                probabilities = torch.nn.functional.softmax(output[0], dim=0)

                top_prob, top_id = torch.topk(probabilities, 2)
                if top_id[0].item() == classOfFrame:
                    correctPredictions += 1
                    print(str(top_id[0].item())+str(classOfFrame))
                totalPredictions += 1

        print(str(correctPredictions)+"--"+str(totalPredictions))