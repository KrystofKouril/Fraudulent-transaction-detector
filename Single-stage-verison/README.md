__4/2/2025__<br />
__Overview:__<br />
This iteration of the project trains a HistGradientBoosting model and saves the model as well as all the other important information into separate files in order for the model to be deployed. Despite HistGradientBoosting being good at handling highly imbalanced data I have to some degree employed SMOTE (Synthetic Minority Oversampling Technique) to supplement the training. The training also consists of a threshold selection procedure in order to select the optimal selection threshold that yields the desirable recall and precision. This all is contained in the training part of the algorithm.

The other part of the algorithm applies the trained model on a new previously unseen dataset (assuming the anonymized columns in it are equivalent to the training set).

The last characteristic of this algorithm is the evaluation metrics of the model’s performance on the new data, including an confusion matrix and a precision-recall curve where the area under said curve is of interest when evaluating the performance of the model.

This model manages an approximately 80% recall rate and a somewhat acceptable false positive rate.

__Data:__<br />
The data was obtained by splitting the original dataset from kaggle into a training set of 60% of the size with the rast being used as the new test data. This split was done using a custom algorithm that did the split randomly while maintaing the class ratio same in both of the new files as it was in the original file.

__Development and technicalities:__<br />
I switched from the previous enseble to HistGradientBoosting simply to save training and to be able to use the whole dataset at hand. Also, as mentioned this new model is good at handling highly imbalanced sets like this one.

Only thing I will say that the development process was a delicate balancing act of all the different parameters. I tried ensuring precision and recall through different means(SMOTE rate, learning rate, etc.) I spent about three days doing only that and managed to strike the final state that you can witness now.

The training sequence throws some FutureWarnings I therefore would not recommend running that sequence and only running the inferrence (infer) mode. This is also becazse at least on my device this training took about fourty minutes.
