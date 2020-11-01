import torch
import cv2
import matplotlib.pyplot as plt
import os
import PIL
import numpy as np
import torchvision
import pickle
from dataloader import get_dataloader_CKPlus

np.set_printoptions(precision=3)

class FaceEmoteBlock(torch.nn.Module):
    def __init__(self, input_channels=32):
        super().__init__()
        self.leakyrelu = torch.nn.LeakyReLU(0.1)
        self.conv1a = torch.nn.Conv2d(kernel_size=1, in_channels=input_channels, out_channels=input_channels // 4)
        self.conv3a = torch.nn.Conv2d(kernel_size=3, in_channels=input_channels, out_channels=input_channels, stride=2)
        self.conv5a = torch.nn.Conv2d(kernel_size=5, in_channels=input_channels, out_channels=input_channels, padding=1,
                                      stride=2)
        self.conv3b = torch.nn.Conv2d(kernel_size=3, in_channels=input_channels // 4, out_channels=input_channels // 4,
                                      stride=2)
        self.conv1a = torch.nn.Conv2d(kernel_size=1, in_channels=input_channels, out_channels=input_channels // 4)
        self.conv1b = torch.nn.Conv2d(kernel_size=1, in_channels=input_channels, out_channels=input_channels // 4)
        self.conv1c = torch.nn.Conv2d(kernel_size=1, in_channels=input_channels, out_channels=input_channels // 4)

    def forward(self, x):
        # Path A
        x_1 = self.conv1a(x)
        x_1 = self.conv3b(x_1)
        x_1 = self.leakyrelu(x_1)

        # Path B
        x_2 = self.conv3a(x)
        x_2 = self.leakyrelu(x_2)
        x_2 = self.conv1b(x_2)

        # Path C
        x_3 = self.conv5a(x)
        x_3 = self.leakyrelu(x_3)
        x_3 = self.conv1a(x_3)

        # Concatenation
        c = torch.cat((x_1, x_2, x_3), axis=1)

        return c

class FacEmoteModel(torch.nn.Module):
    def __init__(self, lr=0.005, epochs=150, image_input = 32):
        super().__init__()

        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.epochs = epochs
        self.optim = torch.optim.Adam
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.CKPlusClasses = ['anger','contempt','disgust','fear','happy','sadness','surprise']

        self.expandLayer = torch.nn.Conv2d(kernel_size=1, in_channels=1, out_channels=image_input)
        self.faceEmote1 = FaceEmoteBlock(input_channels=32)
        self.faceEmote2 = FaceEmoteBlock(input_channels=24)
        self.faceEmote3 = FaceEmoteBlock(input_channels=18)
        self.faceEmote4 = FaceEmoteBlock(input_channels=12)
        self.linear = torch.nn.Linear(in_features = 36, out_features = 7)

    def predict_class(self, x):
        out = self(x)
        out = np.argmax(self.logSoftmax(out).cpu().detach().numpy(), axis=1)
        print(f'Out:{out}')
        return self.CKPlusClasses[int(out)]

    def predict_from_image(self, img):
        imtensor = torch.tensor(img, dtype=torch.float, device=self.device)
        imtensor = imtensor.reshape(1, 1, 48, 48)
        output_class = self.predict_class(imtensor)
        return output_class

    def accuracy(self, y_hat, y):
        return torch.sum(y_hat == y).item() / len(y)

    def validation_evaluation(self, valid_dl):
        with torch.no_grad():
            batch_loss = []
            batch_accuracy = []
            batch_counter = 0

            for x,y in valid_dl:
                y = y.reshape(y.shape[0])
                batch_counter += 1
                outputs = self(x)
                valid_loss = self.loss_func(outputs, y)
                y_hat = self.softmax(outputs)
                y_hat = torch.argmax(outputs, axis=1)
                valid_accuracy = self.accuracy(y_hat, y)

                batch_loss.append(valid_loss.cpu().detach().numpy())
                batch_accuracy.append(valid_accuracy)
                #print(f'Valid Batch:{batch_counter}, Loss:{valid_loss}, Accuracy:{valid_accuracy}')

            total_valid_loss = np.mean(batch_loss)
            total_valid_accuracy = np.mean(batch_accuracy)

        print(f'Total Valid Loss:{total_valid_loss}, Total Valid Accuracy:{total_valid_accuracy}')

        return total_valid_loss, total_valid_accuracy

    def forward(self, x):

        x = self.expandLayer(x)
        x = self.faceEmote1(x)
        x = self.faceEmote2(x)
        x = self.faceEmote3(x)
        x = self.faceEmote4(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)

        return x

    def loss_func(self, op, vals):
        return self.loss(op, vals)

    def train(self, training_dl, valid_dl):

        best_validation_accuracy = 0.0
        counter = 0

        loss_history = []
        acc_history = []
        valid_loss_history = []
        valid_acc_history = []
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9,0.999))

        for i in range(self.epochs):
            batch_counter = 0
            batch_loss = []
            batch_accuracy = []

            for x, y in training_dl:
                y = y.reshape(y.shape[0])
                batch_counter += 1
                outputs = self(x)
                training_loss = self.loss_func(outputs, y)
                y_hat = self.softmax(outputs)
                y_hat = torch.argmax(outputs, axis=1)

                training_accuracy = self.accuracy(y_hat, y)

                batch_loss.append(training_loss.cpu().detach().numpy())
                batch_accuracy.append(training_accuracy)

                if batch_counter % 10 == 0:
                    print(f'Epoch:{i}, Batch:{batch_counter}, Loss:{training_loss}, Accuracy:{training_accuracy}')

                training_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss = np.mean(batch_loss)
            epoch_accuracy = np.mean(batch_accuracy)
            valid_loss, valid_acc = self.validation_evaluation(valid_dl)

            valid_loss_history.append(valid_loss)
            valid_acc_history.append(valid_acc)
            loss_history.append(epoch_loss)
            acc_history.append(epoch_accuracy)

            print(f'Epoch:{i}, Training Loss:{epoch_loss}, Training Acc:{epoch_accuracy}, Validation Loss:{valid_loss}, Validation Acc:{valid_acc}')

            if valid_acc > best_validation_accuracy:
                torch.save({
                    'epoch': i,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, './model_checkpoints/{}_checkpoint_{}.pth'.format('ER_Model_Custom', counter))
                counter += 1
                best_validation_accuracy = valid_acc
                print('Best model has been saved at: {}'.format(i))
                print('Best validation accuracy achieved: {:.3f}'.format(best_validation_accuracy))


        return self.epochs, loss_history, acc_history, valid_loss_history, valid_acc_history

    def predict(self, x):
        output = self(x)
        output = self.softmax(output)
        output = torch.argmax(output, axis=1)
        return output

if __name__ == '__main__':
    train_dl, valid_dl, test_dl = get_dataloader_CKPlus()
    ERModel = FacEmoteModel()

    epochs, train_loss, train_acc, valid_loss, valid_acc = ERModel.train(train_dl, valid_dl)
    test_loss, test_accuracy = ERModel.validation_evaluation(test_dl)

    training_dict = {
        'epochs':epochs,
        'train_loss':train_loss,
        'train_acc':train_acc,
        'valid_loss':valid_loss,
        'valid_acc':valid_acc,
        'test_loss':test_loss,
        'test_acc':test_accuracy
    }

    pickle_dump = open("dict.pickle","wb")
    pickle.dump(training_dict, pickle_dump)
    pickle_dump.close()
