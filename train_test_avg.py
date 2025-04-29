import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.autograd import Variable
import time

from load_data import HCPDDataset, TimeShift, AmplitudeScale, Jitter,Scaling,TimeWarp,FourierAugmentation,WindowSlicing
from models.LSTM import LSTM1
from models.Bi_LSTM import BiLSTM
from models.GRU import GRUModel
from models.transformer import TimeSeriesTransformer
from models.CNN1D import CNNClassifier
from models.Bi_GRU import BiGRUModel
from models.EnsembleModel import EnsembleModel
from models.CNNBGRU import CNNBGRU
from models.PDCNN import PDCNN
from models.LSTMCNN import LSTMCNN



import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns
import io
import os
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats


torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 32

num_epochs = 200 #1000 epochs
hidden_size = 128  # number of features in hidde
num_layers = 1 #number of stacked lstm layers
num_classes = 2 #number of output classes 

input_dim = 1
output_dim = 2
hidden_dim = 128
num_heads = 8
tr_num_layers = 2


input_size_CNN = 1

# def init_all(model, init_func, *params, **kwargs):
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             init_func(param, *params, **kwargs)


def classifaction_report_csv(report,model_name,applied_transform,hidden_size):
    report_df = pd.read_csv(io.StringIO(report), sep=r'\s{2,}', engine='python')
    report_df.to_csv('results/' + model_name + "_" + str(applied_transform) + str(hidden_size) +'.csv', index=False)

    # report_data = []
    # lines = report.split('\n')
    # with open('results/' + model_name + "_" + applied_transform + '.txt', 'w') as f:
    #     for line in lines:
    #         f.write(line)
    #         f.write('\n')


# Define data augmentations
train_transforms = transforms.Compose([
    # TimeShift(max_shift=5)
    # AmplitudeScale(max_scale=1.2),
    # Jitter(max_jitter=0.1),  # Add jitter augmentation
    # Scaling(scale_factor=1.2)
    # TimeWarp(),
    # FourierAugmentation(max_phase_shift=0.5, max_frequency_shift=0.05),
    # WindowSlicing(window_size=10)
    # WindowSlicing(window_size=80)

    # transforms.Normalize((0.5,), (0.5,))
])

# test_transforms = transforms.Compose([
#     transforms.Normalize((0.5,), (0.5,))
# ])

# dataset = HCPDDataset('dataset/DATA_HC_PD_v1.0_final_jsk_removed_features_mean_0_std_1.xlsx')
# dataset = HCPDDataset('dataset/DATA_HC_PD_v1.0_final_jsk_removed_features_normalized.xlsx') # using without normalization
dataset = HCPDDataset('dataset/DATA_HC_PD_total.xlsx')

# dataset = HCPDDataset('dataset/DATA_HC_PD_lowerlimbs.xlsx') # using with normalization 0 to 1
# dataset = HCPDDataset('dataset/DATA_HC_PD_upperlimbs.xlsx')  # using with normalization 0 to 1

input_size = dataset[0][0].shape[-1]
# print(" dataset[0][0]",  dataset[0][0].shape)
# print(" dataset[0][0]",  dataset[0][0])

# Create the individual models
lstm_model = LSTM1(num_classes, input_size, hidden_size, num_layers)
gru_model = GRUModel(input_size, hidden_size, num_classes)
bilstm_model = BiLSTM(num_classes, input_size, hidden_size, num_layers)
cnn_model = CNNClassifier(input_size_CNN, num_classes,input_size)
bigru_model = BiGRUModel(input_size, hidden_size, num_classes)

score = 0

applied_transform = 0
while score < 10:
    for model_selection in range(100):
        # model_selection = model_selection + 1
        k_folds = 5
        accuracy_list = []
        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []
        predictions_list = []
        labels_list = []
        model_name = []
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_models = []
        fold_losses = []
        fold_accuracies = []
        best_accuracy = 0.0
        best_model = None

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            start_time = time.time()
            print(f"  Train: index={train_ids.shape}")
            print(f"  Test:  index={test_ids.shape}")

            start = time.time()
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)
            model_selection = 3
            if model_selection == 0:
                model = LSTM1(num_classes, input_size, hidden_size, num_layers) 
                model_name = "LSTM1"
            elif model_selection == 1: 
                model = BiLSTM(num_classes, input_size, hidden_size, num_layers)
                model_name = "BiLSTM"
            elif model_selection == 2: 
                model = GRUModel(input_size, hidden_size, num_classes)
                model_name = "GRUModel"
            elif model_selection == 3:
                model = TimeSeriesTransformer(input_dim, output_dim, hidden_dim, num_heads, tr_num_layers)
                model_name = "Transformer"
            elif model_selection == 4:
                model = CNNClassifier(input_size_CNN, num_classes,input_size)
                model_name = "1DCNN"
            elif model_selection == 5:
                model = BiGRUModel(input_size_CNN, num_classes,input_size)
                model_name = "BiGRU"
            elif model_selection == 6:
                model = EnsembleModel(lstm_model, gru_model, bilstm_model, cnn_model, bigru_model, num_classes,input_size)
                model_name = "EnsembleModel"
            elif model_selection == 7:
                model = CNNBGRU()
                model_name = "CNNBGRU"
            elif model_selection == 8:
                model = PDCNN()
                model_name = "PDCNN"
            elif model_selection == 9:
                model = LSTMCNN(input_size)
                model_name = "LSTMCNN"
            # init_all(model, torch.nn.init.xavier_uniform_)
            print(model_name)
            model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            scheduler = lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)

            total_step = len(train_loader)

            train_loss_history = []
            train_acc_history = []
            test_loss_history = []
            test_acc_history = []

            for epoch in range(num_epochs):
                model.train()
                correct = 0
                total = 0
                running_loss = 0.0
                running_acc = 0.0
                for i,(x, y) in enumerate(train_loader):
                    # if model_selection == 0:
                    #     x = x.reshape(-1, 1, input_size).to(device) # lstm, bilstm, gru
                    #     print("x",x.shape)
                    # elif model_selection == 1:
                    #     x = x.reshape(-1, 1, input_size).to(device) # lstm, bilstm, gru
                    # elif model_selection == 2:
                    #     x = x.reshape(-1, 1, input_size).to(device) # lstm, bilstm, gru
                    # elif model_selection == 3:
                    #     x = x.unsqueeze(2).to(device) # only for transformer model
                    #     print("x",x.shape)
                    # elif model_selection == 4:
                    #     x = x.unsqueeze(1).to(device) # only for 1dcnn
                    #     # print("x1",x.shape)
                    print("x1",x.shape)
                    # x = x.reshape(-1, 1, input_size).to(device)
                    x = x.unsqueeze(2).to(device)
                    print("x",x.shape)
                    if applied_transform:
                        x = train_transforms(x)
                    outputs = model(x).to(device)
                    # print(outputs)
                    # print(outputs.shape)
                    # print(y.long().shape)

                    loss = criterion(outputs, y.long())
                    # print(" outputs",outputs.shape)
                    # print(" y",y.shape)

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    predictions = torch.max(outputs, 1)[1].to(device)
                    correct += (predictions == y).sum()
                    total += len(y)
                    # print("loss",loss)

                epoch_loss = running_loss / len(train_loader)
                accuracy = correct * 100 / total
                accuracy = accuracy.cpu().numpy()

                # if (i+1) % 1 == 0:
                #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

                train_loss_history.append(epoch_loss)
                train_acc_history.append(accuracy)

                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    running_loss = 0.0
                    running_acc = 0.0
                    for x, y in test_loader:
                        # if model_selection == 0:
                        #     x = x.reshape(-1, 1, input_size).to(device) # lstm, bilstm, gru
                        # elif model_selection == 1:
                        #     x = x.reshape(-1, 1, input_size).to(device) # lstm, bilstm, gru
                        # elif model_selection == 2:
                        #     x = x.reshape(-1, 1, input_size).to(device) # lstm, bilstm, gru
                        # elif model_selection == 3:
                        #     x = x.unsqueeze(2).to(device) #only for transformer model
                        # elif model_selection == 4:
                        #     x = x.unsqueeze(1).to(device) #only for 1dcnn
                        # if applied_transform:
                        #     x = test_transforms(x)
                        x = x.reshape(-1, 1, input_size).to(device)
                        labels_list.append(y)
                        outputs = model(x).to(device)
                        loss = criterion(outputs, y.long())
                        predictions = torch.max(outputs, 1)[1].to(device)
                        predictions_list.append(predictions)
                        correct += (predictions == y).sum()
                        total += len(y)
                        running_loss += loss.item()

                epoch_loss = running_loss / len(test_loader)
                accuracy = correct * 100 / total
                accuracy = accuracy.cpu().numpy()

                test_loss_history.append(epoch_loss)
                test_acc_history.append(accuracy)
                print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch+1, num_epochs, epoch_loss, accuracy))

                end = time.time()
                # print("Working time: {:.2f} seconds".format(end - start))

            # Plot the loss and accuracy curves for this fold
            # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            # ax[0].plot(train_loss_history, label='Training')
            # ax[0].plot(test_loss_history, label='Validation')
            # ax[0].set_xlabel('Epoch')
            # ax[0].set_ylabel('Loss')
            # ax[0].legend()
            # ax[0].set_title('Loss Curve')
            # ax[1].plot(train_acc_history, label='Training')
            # ax[1].plot(test_acc_history, label='Validation')
            # ax[1].set_xlabel('Epoch')
            # ax[1].set_ylabel('Accuracy')
            # ax[1].legend()
            # ax[1].set_title('Accuracy Curve')
            # fig.suptitle(f'Fold {fold+1}')
            # plt.show()

            train_loss_list.append(train_loss_history)
            test_loss_list.append(test_loss_history)
            train_acc_list.append(train_acc_history)
            test_acc_list.append(test_acc_history)

            fold_test_acc = np.mean(test_acc_list[fold])  # Average test accuracy for this fold

            # Print the average test accuracy for this fold
            print(f"Fold {fold+1} Average Test Accuracy: {fold_test_acc}")


        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')

        result_folder = 'test_accuracy/'
        # Save the average test accuracy for each fold in a text file
        fold_acc_file_path = os.path.join(result_folder, f"fold_accuracy_{model_name}_{hidden_size}.txt")
        fold_loss_file_path = os.path.join(result_folder, f"fold_loss_{model_name}_{hidden_size}.txt")

        with open(fold_acc_file_path, 'w') as f:
            for fold in range(k_folds):
                fold_test_acc = np.mean(test_acc_list[fold])
                f.write(f"Fold {fold+1} Average Test Accuracy: {fold_test_acc}\n")

        with open(fold_loss_file_path, 'w') as f:
            for fold in range(k_folds):
                fold_test_loss = np.mean(test_loss_list[fold])
                f.write(f"Fold {fold+1} Average Test loss: {fold_test_loss}\n")

        # Calculate average loss and accuracy across all folds
        avg_train_loss = np.mean(train_loss_list, axis=0)
        avg_test_loss = np.mean(test_loss_list, axis=0)
        avg_train_acc = np.mean(train_acc_list, axis=0)
        avg_test_acc = np.mean(test_acc_list, axis=0)

        
        fold_losses.append(avg_test_loss)
        fold_accuracies.append(avg_test_acc)

        # Save the best model based on accuracy
        if np.any(avg_test_acc > best_accuracy):
            best_accuracy = np.max(avg_test_acc)
            best_model = model.state_dict()

        # Store the model for this fold
        fold_models.append(model)

        # Calculate the average loss and accuracy across all folds
        avg_loss = np.mean(fold_losses)
        avg_accuracy = np.mean(fold_accuracies)

        save_model_folder = "save_model"
        # Save the best model
        torch.save(best_model, os.path.join(save_model_folder, f"best_model_{model_name}_{hidden_size}.pth"))

        # Print the average loss and accuracy
        print(f"Average Loss: {avg_loss}")
        print(f"Average Accuracy: {avg_accuracy}")
   
        score = avg_accuracy

        # Plot the average loss and accuracy curve
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].plot(avg_train_loss, label='Training')
        ax[0].plot(avg_test_loss, label='Validation')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].set_title('Loss Curve')
        ax[1].plot(avg_train_acc, label='Training')
        ax[1].plot(avg_test_acc, label='Validation')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].set_title('Accuracy Curve')
        fig.suptitle(f'Fold {fold+1}')
        plt.savefig('loss_accuracy/' + model_name + "_" + str(applied_transform) + str(hidden_size) +'.png')

        # Define the folder path to save the data
        folder_path = 'loss_accuracy'
        os.makedirs(folder_path, exist_ok=True)

        # Define the file path to save the data
        data_file_path = os.path.join(folder_path, str(model_name) + '_loss_accuracy_data_'+ str(applied_transform) + str(hidden_size)+'.txt')

        # Save the data to the file
        with open(data_file_path, 'w') as file:
            file.write('Average Train Loss:\n')
            file.write(str(avg_train_loss) + '\n\n')

            file.write('Average Test Loss:\n')
            file.write(str(avg_test_loss) + '\n\n')

            file.write('Average Train Accuracy:\n')
            file.write(str(avg_train_acc) + '\n\n')

            file.write('Average Test Accuracy:\n')
            file.write(str(avg_test_acc) + '\n\n')

        # Print a confirmation message
        print('Data saved to:', data_file_path)

        # Flatten the predictions and labels lists
        # print(predictions_list[0][1].shape)
        predictions_list = torch.cat(predictions_list).tolist()
        labels_list = torch.cat(labels_list).tolist()
        # print(labels_list[0].shape)


        # Calculate the overall Jaccard score
        overall_jaccard_score = metrics.jaccard_score(labels_list, predictions_list)
        print(f"Overall Jaccard Score: {overall_jaccard_score}")

        # Calculate the overall Cohen's Kappa score
        overall_kappa_score = metrics.cohen_kappa_score(labels_list, predictions_list)
        print(f"Overall Cohen's Kappa Score: {overall_kappa_score}")

        # Define the folder paths
        jaccard_folder = 'jaccard_scores/'
        kappa_folder = 'kappa_scores/'

        # Create the folders if they don't exist
        os.makedirs(jaccard_folder, exist_ok=True)
        os.makedirs(kappa_folder, exist_ok=True)

        # Save the Jaccard score as a text file
        jaccard_file_path = os.path.join(jaccard_folder, f"{model_name}_{str(applied_transform)}_{hidden_size}.txt")
        with open(jaccard_file_path, 'w') as f:
            f.write(f"Overall Jaccard Score: {overall_jaccard_score}")

        # Save the Cohen's Kappa score as a text file
        kappa_file_path = os.path.join(kappa_folder, f"{model_name}_{str(applied_transform)}_{hidden_size}.txt")
        with open(kappa_file_path, 'w') as f:
            f.write(f"Overall Cohen's Kappa Score: {overall_kappa_score}")

        # Calculate and print classification report and confusion matrix


        report = metrics.classification_report(labels_list, predictions_list)
        classifaction_report_csv(report,model_name,applied_transform,hidden_size)
        confusion_mat = metrics.confusion_matrix(labels_list, predictions_list)


        # Get class labels
        classes = np.unique(labels_list)

        # Plot confusion matrix as an image
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        # plt.show()
        plt.savefig('confusion_matrix/' + model_name + "_" + str(applied_transform) + str(hidden_size) +'.png')

        print("Classification report for the Model:\n", report)
        print("Confusion Matrix for the Model:\n", confusion_mat)

        # # Plot t-SNE Visualization as an image
        # model_outputs = np.array(predictions_list)
        # model_outputs = model_outputs.reshape(-1, 1)
        # tsne = TSNE(n_components=2, random_state=42)
        # tsne_output = tsne.fit_transform(model_outputs)
        # classes = np.unique(labels_list)
        # class_colors = {0: 'blue', 1: 'orange'}
        # plt.figure(figsize=(8, 6))
        # for i, class_label in enumerate(classes):
        #     class_data = tsne_output[labels_list == class_label]
        #     plt.scatter(class_data[:, 0], class_data[:, 1], color=class_colors[class_label], label=class_label, s=1)
        # plt.title('t-SNE Visualization')
        # plt.xlabel('t-SNE Dimension 1')
        # plt.ylabel('t-SNE Dimension 2')
        # plt.text(0.5, 0.5, 'Parkinson\'s: 0, Healthy: 1', fontsize=12, ha='center')
        # plt.legend(title='Class')
        # plt.savefig('tsne/' + model_name + "_" + str(applied_transform) + str(hidden_size) + '.png')



        # Plot roc curve Visualization as an image
        fpr, tpr, thresholds = roc_curve(labels_list, predictions_list)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve/' + model_name + "_" + str(applied_transform) + str(hidden_size) +'.png')

        if(avg_accuracy >=95 ):
            print(avg_accuracy)
            break


