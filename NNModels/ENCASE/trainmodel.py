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
print("Root:", root)

if not os.path.exists(root):
    exit()
if not os.path.exists(os.path.join(root, "Folds")):
    print("Check again")
    exit()

from extract_signals import *
from resnet import (
    linear_feature_extractor_classifier,
    ResNet1D,
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
    start_epoch=0,
    best_score=0,
):
    best = f"{root}trainedModels/fold{current_fold}_{name}.pth"
    checkpoint_path = f"{root}trainedModels/fold{current_fold}_{name}_checkpoint.pth"
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # if exists, just return the path
    if os.path.exists(best):
        print("Model already trained")
        return best

    curr_best = best_score
    curr_name = ""
    step = 0
    pbar = tqdm(range(start_epoch, n_epoch), desc="epoch")
    for epoch in pbar:
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

        # 更新进度条的后缀信息
        pbar.set_postfix(F1=score)

        # update the best model when necessary
        if score > curr_best:
            try:
                os.remove(curr_name)
            except:
                pass
            curr_best = score
            curr_name = f"{root}trainedModels/fold{current_fold}_{name}.pth"
            torch.save(model.state_dict(), curr_name)

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_score': curr_best,
        }, checkpoint_path)

    return curr_name


def train_fold(
    current_fold,
    root,
    num_cores=8,
    n_epoch1=50,
    n_epoch2=50,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
):
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

    dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

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

    current_best = train_and_validate(
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

    model.load_state_dict(torch.load(current_best))

    model.classifier = linear_feature_extractor_classifier(
        1024, 4
    )  # change the classifier to a new one
    model.to(device)

    # 重新初始化优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    print("Training the second stage")

    best_model_path = train_and_validate(
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

    test_expanded = multi_features(X_test_raw, n_cores=num_cores)

    X_train = np.expand_dims(train_expanded, 1)
    X_test = np.expand_dims(test_expanded, 1)
    y_train = y_train_raw

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, np.zeros(X_test.shape[0]))
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

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
    model.load_state_dict(torch.load(best_model_path))
    model.dense = torch.nn.Identity()
    model.to(device)
    model.eval()

    def get_features(dataloader):
        features = []
        for batch_idx, batch in enumerate(dataloader):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            n_pred = pred.cpu().data.numpy()
            features += [x for x in n_pred]
        return features

    train_features = np.array(get_features(train_dataloader))
    test_features = np.array(get_features(test_dataloader))
    print("Features extracted")
    print(train_features.shape, test_features.shape)

    np.savetxt(
        root + "Data/features/" + f"resnet_training_features{current_fold}.txt",
        train_features,
        delimiter=",",
    )
    np.savetxt(
        root + "Data/features/" + f"resnet_test_features{current_fold}.txt",
        test_features,
        delimiter=",",
    )


def main(test=True, batch_size=128, num_workers=4, pin_memory=True):
    if test:
        train_fold(
            current_fold, root, num_cores=8, n_epoch1=1, n_epoch2=1
        )  # run for only one fold and one epoch
    else:
        for i in range(5):
            train_fold(
                i,
                root,
                num_cores=16,
                n_epoch1=50,
                n_epoch2=50,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )


if __name__ == "__main__":
    print("Testing for only one fold and one epoch")
    main()
