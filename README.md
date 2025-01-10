SMS-SPAM-DETECTION-USING-NPL Overview The project discusses spam message detection in SMS, using Natural Language Processing. We are going to build a machine learning model that will classify SMS messages into "spam" or "ham". We intend to leverage multiple NLP techniques such as preprocessing of text, feature extraction, and machine learning algorithms to identify spam messages and filter them out.
This paper will be using the SMS Spam Collection Dataset. It is the collection of various SMS messages categorized either as "spam" or "ham." Each one of them is shown with one line string text and associated with a class label: Spam: Unsolicited, irrelevant, or unwanted messages, such as advertisements or phishing attempts. Ham: These are legitimate messages, for instance, a personal message, question, or job-related messages. The dataset comes in two parts:

Training data are those that the system uses to train the machine learning model. Testing data is for measuring the performance of a model. **Requirements Used in this project, the Python libraries required are:**

pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
tensorflow (if using deep learning models)
Steps Involved
1. Preprocessing
Data Loading: Import the dataset into a Pandas DataFrame.
Cleaning: Remove irrelevant characters such as special symbols, digits, and extra whitespaces.
Text Normalization: All texts should be made to lowercase to make the input consistent.
Stopword removal: The common words like "the," "and" should be removed since these do not add much value in classifying a message. Tokenization: Split the text or review into individual words (tokens). 2. Feature Extraction Bag-of-Words BoW: Representation of text data as a bag of words and their frequencies. TF-IDF: Weighting the words based on importance in the entire dataset. Word Embedding: Representing words as vectors using pre-trained models such as Word2Vec, GloVe. 3. Model Training
Train various machine learning models such as Logistic Regression, Naive Bayes, Support Vector Machine, Random Forest, and Deep Learning model Neural Networks. 4. Model Evaluation Accuracy: Calculate the ratio of correct classification of messages. Precision, Recall, F1-score: Check the performance of the model concerning both spam and ham classes. Confusion Matrix: Draw the count of TP, FP, TN, and FN instances. 5. Final Model
Finally, after training and testing the model, it can be saved for later predictions via serialization techniques, for example, joblib or pickle. Example Usage To run the training and testing of the spam detection model, the following command could be executed: bash Copy code python main.py This script will pre-process the data, train the model, and report some performance metrics. Future Improvements Deep Learning Models: Utilize higher deep learning models such as LSTM or BERT that can provide higher accuracy.
Real-Time Detection: Integrate the model with some messaging application for real-time spam detection.
User Feedback Loop: Provide a way for users to give feedback on false positives/negatives to update the model.
References
SMS Spam Collection Dataset
Scikit-learn Documentation
NLTK Documentation
