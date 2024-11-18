import pandas as pd
import numpy as np

import torch
import os
import sys
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader


current_folder = os.path.dirname(os.path.realpath(__file__))

root = "../../"
sys.path.append(current_folder)
print(os.path.join(current_folder, root))
sys.path.append(os.path.join(current_folder, root))
root = os.path.join(current_folder, root)
print(root)

if not os.path.exists(root):
    exit()
if not os.path.exists(os.path.join(root, "Folds")):
    print("Check again")
    exit()

from extract_signals import *
from resnet import (
    linear_feature_extractor_classifier,
    ResNet1D,
    linear_feature_extractor_classifier,
    MyDataset,
)
from utils import *


current_fold = 0
device_str = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device_str = "cuda"


def train_and_validate(
    model,
    dataloader,
    val_dataloader,
    loss_func,
    optimizer,
    scheduler,
    device,
    n_epoch,
    current_fold,
    root,
    name,
):
    curr_best = 0
    curr_name = ""
    step = 0
    for epoch in range(n_epoch):
        # 训练
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        scheduler.step(epoch)

        # 验证
        model.eval()
        all_pred_prob = []
        true_pred = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                all_pred_prob.append(pred.cpu().data.numpy())
                true_pred.append(input_y.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        all_true_pred = np.concatenate(true_pred)
        score = f1_score(all_true_pred, all_pred, average="micro")
        print("Fold:", current_fold, "Epoch:", epoch, "F1:", score)

        # update the best model when necessary
        if score > curr_best:
            curr_best = score
            curr_name = f"{root}trainedModels/fold{current_fold}_{name}.pth"

        if (epoch + 1) % 10 == 0 or epoch == n_epoch - 1:  # periodically save the model
            # periodically save the model
            try:
                os.remove(curr_name)
            except:
                pass
            torch.save(model.state_dict(), curr_name)


def train_fold(current_fold, root, num_cores=8, n_epoch1=50, n_epoch2=50):
    X_train_raw, y_train_raw, X_test_raw = load_Xtrain_ytrain_Xtest()
    y_train_raw = y_train_raw.to_numpy()

    # expand the train to 6000 features

    os.environ["OMP_NUM_THREADS"] = "1"

    train_expanded = multi_features(X_train_raw, n_cores=num_cores)

    import json

    def load_file(name, index):
        path = os.path.join(root, "Folds/")
        with open(path + name + str(index) + ".json") as f:
            arr = json.load(f)

        return arr

    train_indices = np.array(load_file("train", current_fold))
    val_indices = np.array(load_file("val", current_fold))

    X_train = train_expanded.iloc[train_indices]
    X_val = train_expanded.iloc[val_indices]
    y_train = y_train_raw[train_indices]
    y_val = y_train_raw[val_indices]
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)

    print(X_train.shape, y_train.shape)

    batch_size = 32

    dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    kernel_size = 16
    stride = 2
    n_block = 48
    downsample_gap = 6
    increasefilter_gap = 12
    # classifier = linear_classifier(1024, 4)
    classifier = linear_feature_extractor_classifier(1024, 4)

    model = ResNet1D(
        in_channels=1,
        base_filters=128,  # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=kernel_size,
        stride=stride,
        groups=32,
        n_block=n_block,
        n_classes=4,
        downsample_gap=downsample_gap,
        increasefilter_gap=increasefilter_gap,
        use_do=True,
        classifier=classifier,
    )

    model.to(device)
    print(X_train.shape[1], X_train.shape[2])

    import torch.optim as optim

    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )  # set up scheduler, it will reduce the learning rate by 0.1 if the loss does not decrease for 10 consecutive epochs
    loss_func = torch.nn.CrossEntropyLoss()

    train_and_validate(
        model,
        dataloader,
        val_dataloader,
        loss_func,
        optimizer,
        scheduler,
        device,
        n_epoch1,
        current_fold,
        root,
        "resnet_classifier",
    )

    current_best = f"{root}trainedModels/fold{current_fold}_resnet_classifier.pth"
    model.load_state_dict(torch.load(current_best))

    model.classifier = linear_feature_extractor_classifier(
        1024, 4
    )  # change the classifier to a new one
    model.to(device)

    print("Training the second stage")

    train_and_validate(
        model,
        dataloader,
        val_dataloader,
        loss_func,
        optimizer,
        scheduler,
        device,
        n_epoch2,
        current_fold,
        root,
        "resnet_extractor",
    )

if __name__ == "__main__":
    train_fold(current_fold, root, num_cores=8, n_epoch1=1, n_epoch2=1)