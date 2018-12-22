Problem Set 5

1. segment_and_tokenize splits the string into sentences. Each sentence is split
up by word and put into a list, in order. Each sentence list is put into a larger
single list. That larger list is the one that is returned. If the sentence contains
a word that does not appear a lot in the input, those words are replaced by the 
string <UNKOWN> (this is done in the method remove_infrequent_words).

2. make_to_ix takes each of the unique words in the sentences and assigns a unique 
integer value to it. It returns a dictionary: the keys are the words and the values 
are unique numbers.

3. vectorize_sents gets the one-hot vectors by using the dictionary that was created
in the method make_to_ix. Each of the words in the sentence is converted to a one-hot
vector using the method sent_to_onehot_vecs. Using the dictionary from make_to_ix, we
get a unique number from each word. That index in the list is 1. All other indices in
the array are 0. That list is added to a list along with all the other words in the
list. Then that sentence list is added to another list of sentence lists and that final
list is what is returned.

4. To get a word embedding, the program multiplies the one-hot vector of the word with
a word embedding weight matrix. The weight matrix is created with the
intialize_weight_matrix method. The matrix is initialized to random values. Since the
two input tensors are one-dimensional, this returns a scalar. That scalar is the word
embedding of that particular word.

5. The function Elman unit calculates each element of the matrix with itself when using 
torch.matmul, doing the actual matrix multiplication. We are able to take the sigmoid 
function of W_x and word_embedding multiplied plus the W_h and h_previous multiplied plus b.

6. single_layer_perceptron takes the current hidden state (just ouputted by the RNN) and
multiplies it by the weights (the matric W_p) of the single layer perceptron. That
result is taken and put into the softmax.

7. The loss function converges to about 10000 and stays around that range. One problem 
the current RNN could have is the vanishing gradient problem. This problem causes the 
gradient to be so small that it is unable to change the values of the weights, which 
could prevent the RNN form effectively training. 
