import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


matplotlib.use("TkAgg")


def plot_accuracy_and_loss(history):
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]

    loss = history["loss"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.savefig("Accuracy-Loss-Plot.png")


def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.savefig("Confusion-Matrix")


def plot_class_report(clf_report):
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="Blues")
    plt.savefig("Classification-Report")
