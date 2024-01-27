import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

def test_model(compiled_model: tf.keras.Sequential, test_data) -> None:
    '''
    Method tests the trained model and stores the results.
    '''
    loss, accuracy, true_positives, true_negatives, false_positives, false_negatives = compiled_model.evaluate(test_data)
    print(loss, accuracy, true_positives, true_negatives, false_positives, false_negatives)

    images, labels = test_data.as_numpy_iterator().next()
    predictions = compiled_model.predict_on_batch(images).flatten()

    converted_predictions = tf.where(predictions < 0.5, 0, 1).numpy()

    num_images = len(images)
    _, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i] / 255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"True: {labels[i]}\nPred: {converted_predictions[i]}")
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Printing confusion matrix
    # https://github.com/philipco/time_series_recognition/blob/2fca8e43d659314a7ba68199647c05b0f1e41dc3/plotting/plot_classification_quality.py
    confusion_matrix = np.array([[true_positives, false_positives], [false_negatives, true_negatives]])
    
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    
    plt.title('Confusion Matrix')
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    
    class_names = ['Positive', 'Negative']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.colorbar()
    plt.show()

    # False/True rate visualization
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, predictions)

    _, ax = plt.subplots()
 
    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(false_positive_rate, true_positive_rate)
    
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')
    
    ax.legend(loc='best')
    
    plt.show()

    # Recall visualization
    precision, recall, _ = precision_recall_curve(labels, predictions)
    _, ax = plt.subplots()
 
    ax.plot(precision, recall)
    
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('Precision-Recall curve')
    
    ax.legend(loc='best')
    
    plt.show()