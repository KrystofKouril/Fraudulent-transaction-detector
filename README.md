# Fraudulent-transaction-detector
26/12/2024<br />
__Overview:__<br />
This project is a fraudulent transaction detection algorithm. The algorithm preprocesses the data set (from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data ) using pandas and then uses an ensemble of three machine learning models using Scikit – Random Forrest, Logarithmic regression, and SVM. The goal was not to make the most efficient or accurate program possible but rather to familiarize myself with a range of machine learning algorithms and their application in FinTech. The models have therefore been selected either based on my previous experience with them (logarithmic regression) or simply on them being interesting to me (the other two).<br />

Furthermore, the script contains an unused function that individually evaluates each of the models on the training data set and selects the most accurate one using GridSearch with cross-validation to train and use on the testing dataset.<br />

The output of this program is is an performance evaluation table either for the ensemble model in the primary function or the selected model in the unused function. The program is therefore self-sufficient and with each run if the code it trains itself again. This also means that the only application of this program as it is is to be used for insight into the accuracy of the chosen models and not for actual deployment on other datasets.<br />

Saving the trained model and deploying it on new datasets is a matter of future development which I would like to get to some day.<br />

__Development process and considerations:__<br />
The initial idea was to preprocess the data and then evaluate each of the models individually and then train one of them as way to gain most insight into each of the models which is exactly what the unused function achieves. However after completing set function I encountered complications regarding the computational power of my computer as the evaluating each model (especial Random Forrest) was very demanding and I could use only about 5 % of the dataset for the program to compile in reasonable amount of time.<br />

I have therefore abandoned this version and decided to work with an ensemble of all of them. This largely improved the accuracy but mainly performance as it now allowed me to work with about half of the dataset (over 140 thousand transactions).<br />

I have encountered many other complications in the development process, mainly regarding the data itself. In general any data set of transactions containing some fraudulent ones is going to be extremely unbalanced as there are significantly less fraudulent than legitimate transactions. We are therefore focusing on the minority class which dictates some parts of the model. It necessitates the use of class weights due to the extreme imbalance of the data classes. They are automatically applied in the newer function (the ensemble model function), however I have somewhat wastefully encoded a condition evaluating whether the set is unbalanced and based on that the application of class weights in the initial function. In that case I set the threshold for what is an unbalanced set quite high as the the goal of the program is to be as accurate as possible in the minority class.<br />
