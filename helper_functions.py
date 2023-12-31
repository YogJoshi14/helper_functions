# Created By: Yogesh Joshi
# Github : @yogjoshi14

# Note: The following confusion matrix code is a remix of Scikit-Learn's
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import zipfile
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import datetime
from sklearn.metrics import classification_report
import pandas as pd
import PIL
# Our function needs a different name to sklearn's plot_confusion_matrix


def convert_to_wandb_table(dataframe, table_name, wandb_project, wandb_api_key):
    '''
    Args:
    dataframe, dandb table name, wandb project name, wandb_api_key
    '''
    import wandb
    
    # Initialize a new wandb Table
    # Initialize wandb and log the CSV file as an artifact
    wandb.init(project=wandb_project, api_key=wandb_api_key)
    table = wandb.Table(columns=dataframe.columns.tolist(), name=table_name)

    # Convert DataFrame rows to Table rows
    for index, row in dataframe.iterrows():
        table_row = [row[column] for column in dataframe.columns]
        table.add_data(*table_row)

    wandb.log({table_name: table})
    wandb.finish()


def add_artifacts_wandb(filename,wandb_project,wandb_api_key,afrifact_name,artifact_data_type):
    '''
    Args:
    local file, project_name, api_key, save_artifact name, artifact datatype
    '''
    
    import wandb
    # Initialize wandb and log the CSV file as an artifact
    wandb.init(project=wandb_project, api_key=wandb_api_key)
    artifact = wandb.Artifact(afrifact_name, type=artifact_data_type)
    artifact.add_file(filename)
    wandb.log_artifact(artifact)

    # Finish the run
    wandb.finish()


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False, wandb=False):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).
      wandb: wandb object.
    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / \
        cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    # colors will represent how 'correct' a class is, darker == better
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           # create enough axis slots for each class
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           # axes will labeled with class names (if they exist) or ints
           xticklabels=labels,
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)
    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")

    #To add cm to wandb 
    if wandb:
        # wandb.init(project=wandb_project, api_key=wandb_api_key)
        wandb.log({"confusion_matrix": wandb.Image(plt)})




def make_classification_report(y_true, y_pred, test_dir_list, pred_probs, class_names=None):
    '''
    Inputs : labels, predictions, list of class names, report_name to be saved

    Args:
    Ground Truth, Prediction, List of file path, prediction probability, unique_classes list
    '''

    pred_df = pd.DataFrame({"img_path": test_dir_list,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "pred_conf": pred_probs.max(axis=1), # get the maximum prediction probability value
                        "y_true_classname": [class_names[i] for i in y_true],
                        "y_pred_classname": [class_names[i] for i in y_pred]}) 
    
    pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
    
    # Get a dictionary of the classification report
    classification_report_dict = classification_report(y_true, y_pred, output_dict=True)
    # Create empty dictionary
    class_f1_scores = []
    class_precision = []
    class_recall = []
    # Loop through classification report items
    for k, v in classification_report_dict.items():
        if k == "accuracy": # stop once we get to accuracy key
            break
        else:
            # Append class names and f1-scores to new dictionary
            class_f1_scores.append(v["f1-score"])
            class_precision.append(v["precision"])
            class_recall.append(v["recall"])

    cm_df = pd.DataFrame({"class_names":class_names,"precision":class_precision,"recall":class_recall})

    return pred_df,cm_df


# Create function to unzip a zipfile into current working directory
# (since we're going to be downloading and unzipping a few files)


def unzip_data(filename):
    """
    Unzips filename into the current working directory.

    Args:
      filename (str): a filepath to a target zip folder to be unzipped.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()


# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.

    Args:
      dir_path (str): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Function to evaluate: accuracy, precision, recall, f1-score


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array

    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results

    
def plot_some_negative_examples(df,dandb=False):
    top_100_wrong = df[df["pred_correct"] == False].sort_values("pred_conf", ascending=False)[:10]
    top_100_wrong.head(20)
        # 5. Visualize some of the most wrong examples
    images_to_view = 9
    start_index = 10 # change the start index to view more
    plt.figure(figsize=(15, 10))
    for i, row in enumerate(top_100_wrong[start_index:start_index+images_to_view].itertuples()): 
        plt.subplot(3, 3, i+1)
        img = pil_image = PIL.Image.open(row[1])
        _, _, _, _, pred_prob, y_true, y_pred, _ = row # only interested in a few parameters of each row
        plt.imshow(img)
        plt.title(f"actual: {y_true}, pred: {y_pred} \nprob: {pred_prob:.2f}")
        plt.axis(False)

    if wandb:
        import wandb
        # wandb.init(project=wandb_project, api_key=wandb_api_key)
        wandb.log({"Max_False_Positive": wandb.Image(plt)})