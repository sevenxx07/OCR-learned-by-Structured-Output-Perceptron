# Problem definition
The inputs are a binary images containing a line of text. All characters have a fixed width so
that we do not have to solve the segmentation problem. 
I implemented three linear classifiers for sequence recognition, each using a
different model of the sequences. The aim is to learn parameters of the three classifiers
by the Perceptron algorithm and to evaluate their performance.

The input is a binary image I = (I1, . . . , IL) ∈ {0, 1}16×8·L displaying a sequence of L hand-
written characters. Each sub-image Ii ∈ {0, 1}16×8, i ∈ {1, . . . , L}, is of fixed size hence the
segmentation of the image I into characters is known. We represent the input image I by a
matrix X = (x1, . . . , xL) ∈ Rd×L where d is the number of features. The i-th column xi ∈ Rd
is a feature description of the i-th sub-image Ii. The feature vector xi contains the intensity
values of Ii and un-ordered products of the intensity values computed from all different pixel
pairs of Ii. Therefore the number of features is d = 16 · 8 + (16 · 8 − 1)(16 · 8)/2 = 8256. In
addition, each feature vector is normalized to have a unit L2-norm.
The task is to recognize a sequence of characters y = (y1, . . . , yL) ∈ AL, A = {a, b, . . . , z},
depicted on the image I using the features X. Below we describe three different classification
strategies and a way how to evaluate their performance.

# Problem divination
 * Data Loading
 For evaluation and testing, datasets were loaded for each class. The input data consisted of
 images or sequences, and corresponding labels were used for training and validation.<br />
 * Training
 All three classes underwent a training phase. The training process involved updating weights
 and biases based on the provided input features and labels. Each class implemented a specific
 training strategy suitable for its classification task.<br />
 * Prediction
 After training, the models were evaluated on test datasets to assess their prediction accuracy.
 The prediction process varied for each class, reflecting the unique characteristics of the
 implemented algorithms.

# Independent multi-class classifier
 This class was implemented to handle multi-class classification problems using a
 perceptron-based approach. The model computes predictions for each letter separately.<br />
 Advantages:<br />
 * Simple and easy to implement.
 * Performance on sequences that are not in dataset will have approximately similar
 error rates
 * Well-suited for scenarios with a large number of classes.
 * Fasttraining time.<br />
 
 Disadvantages:<br />
 * Ignores correlations between classes.
 * Maynotperform well when classes are highly imbalanced.<br />
Given a feature representation X = (x1, . . . , xL) of an input image I, the characters are
predicted for each sub-image independently by a multi-class linear classifier.<br />
![obrazek](https://github.com/user-attachments/assets/17bea397-1272-4dc4-947c-43e19a58a7cc)<br />
where the parameters are the character templates w = (w1, . . . , w|A|) ∈ Rd·|A| and biases
b = (b1, . . . , b|A|) ∈ R|A|. Note that the length L of the unknown sequence can be deduced from
the width of the input image I or the feature matrix X, respectively.

#  Structured, pair-wise dependency
 This class was designed for structured prediction tasks, where the output is a sequence. The
 model employed a pairwise approach, considering dependencies between adjacent elements
 in the sequence. This class used dynamic programming for efficient sequence decoding
 during prediction.<br />
 Advantages:<br />
 * Captures pairwise dependencies, improving sequence prediction accuracy.
 * Suitable for tasks where the order of elements matters.
 * Performance on sequences that are in the same language will have approximately
 similar error rates<br />

 Disadvantages:<br />
 * Training and predicting sequences may be computationally expensive for long
 sequences.
 * Sensitivity to noise between pairwise dependencies.
 * notideal and for sequences which are in a different language<br />
 The characters are predicted by a linear classifier<br />
![obrazek](https://github.com/user-attachments/assets/09be4b19-7311-426f-87c3-d55c7c84a4dd)<br />
where the parameters are w = (w1, . . . , w|A|) ∈ Rd·|A|, b = (b1, . . . , b|A|) ∈ R|A| and the pair-
wise dependency function g : A × A → R.
Evaluating the predictor leads to a discrete optimization problem which can be solved in
time O(|A|2 · L) by dynamic programming. By introducing a shortcut qi(yi) = 〈wyi , xi〉 + byi ,
we can rewrite it as<br />
![obrazek](https://github.com/user-attachments/assets/d8334b9a-1f59-463d-9cae-241a48212cae)<br />

# Structured, fixed number of sequences
 This class extended the structured prediction concept by handling fixed-length sequences. It
 incorporated weights and biases to capture relationships between input features and output
 sequences. The model was trained using a combination of input features, correct labels, and
 predicted labels.<br />
 Advantages:<br />
 * Handles fixed-length sequences efficiently.
 * Bestperformance
 * Accounts for relationships within sequences through weights and biases.<br />
 
 Disadvantages:<br />
 * Maynotgeneralize well to sequences of very varying lengths.
 * Cannot be used on sequences, which are not in dataset<br />
 Let us assume that the set of hidden sequences Y ⊂ A∗ contains a small number of elements.
For example, in our application Y contains just 20 names (see Figure 2). In this case we can
predict the sequences by a linear classifier<br />
![obrazek](https://github.com/user-attachments/assets/793392d0-05c8-4861-8cd3-3eebe26ae5ed)<br />
where YL ⊂ Y contains all sequences of the length L. The classifier is parametrized by w =
(w1, . . . , w|A|) ∈ Rd·|A|, b = (b1, . . . , b|A|) ∈ R|A| and a function v : Y → R.

# Error measurements 
Let {(X1, y1), . . . , (Xm, ym)} be a set of examples and let {ˆy1, . . . , ˆym} be the predictions
produced by a classifier applied on the inputs {X1, . . . , Xm}. To evaluate a performance of a
classifier we are going to use two error measures. First, the sequence prediction error defined as<br />
![obrazek](https://github.com/user-attachments/assets/273977c8-5cdc-4223-aff8-b31d7258af70)<br />

which is an estimate of the probability that the predicted sequence is not entirely correct.
Second, the character prediction error is defined as<br />
![obrazek](https://github.com/user-attachments/assets/e38b28ba-e3c1-47f4-bfcf-74263b088b75)<br />

where M is the total number of characters in all sequences and Lj is the length of
the j-th sequence. The value of Rchar is an estimate of the probability that a single character
in the sequence is incorrectly classified.

# Data
Data are not uploaded here because I have no rights to do it.

# Results
 In this table are represented Test errors in % of perceptrons.
|  | Rseq   | Rchar  |
| :-----: | :---: | :---: |
| independent multi-class classifier | 70,40 |  26,44 |
| structured, pair-wise dependency | 11,20|  5,43 |
| structured, fixed number of sequences|  1,80   | 1,32  |


