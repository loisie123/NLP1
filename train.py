import torch
from torch.autograd import Variable
import torch.nn as nn


# dit heb ik gekopieerd.. 
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)#, 2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        
        # willen wij dit anders? Maakt hij niet zelf al die hidden states aan...
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores



#>>> rnn = nn.LSTM(10, 20, 2)
#>>> input = Variable(torch.randn(5, 3, 10))
#>>> h0 = Variable(torch.randn(2, 3, 20))
#>>> c0 = Variable(torch.randn(2, 3, 20))
#>>> output, hn = rnn(input, (h0, c0))


#dtype = torch.FloatTensor

def create_folds(X, k):
    """
    Create k folds of equal size for data set X.
    X is a list and k is a non-zero integer.
    """
    N = len(X)
    if k > N: k = N # set k to N if k if bigger than the size of data set
    fold_size = round(N/k)
    folds = []
    for i in range(k):
        train_set = X[:i*fold_size] + X[:(i+1)*fold_size]
        test_set = X[i*fold_size:(i+1)*fold_size]
        folds.append([train_set, test_set])
    return folds


def train(train_data, lr, iterations, layer_info, voca_size, tag_size):
    """
    Create neural netwerk and train it.
    Input:
        :param train_data: nxm np.array
        :param lr
        :param iterations:
        :layer_info
    Output:
        neural netwerk
    """    
    # input data is matrix met als rijen de vector word embedding en postag embedding onder elkaar
    # output data is matrix met als rijen number incoming arc en vector label embedding onder elkaar geplakt.
    
    input_size = 125 # example # lengte (word embedding vector) + lengte (POS TAG embedding vector)
    output_size = 21 # example # lengte (label embedding vector) + 1
    # in this case len(train_data[i]) = 125+21
    hidden_size = 100 # number of nodes in hidden layers
    #num_layer = 2 # number of hidden layers
    
    lstm_net = LSTMTagger(input_size, hidden_size, voca_size, tag_size)
    # lstm_net = nn.LSTM(input_size, hidden_size, num_layer)
    
    # make data ready for use
    in_data = []
    out_data = []
    for matrix in train_data:
        for row in matrix:
            # afhankelijk van hoe Koen de data in de matrix zet...
            in_data.append(row[:125])
            out_data.append(row[125:])
    # ze gebruiken ergens LongTensor ipv FloatTensor.. is dat beter?
    train_input = Variable(torch.FloatTensor(in_data).type(dtype), requires_grad=False)
    train_target = Variable(torch.FloatTensor(out_data).type(dtype), requires_grad=False)
    #train_input = Variable(torch.FloatTensor(train_data[:,:3]).type(dtype), requires_grad=False)
    #train_target = Variable(torch.FloatTensor(train_data[:,-1]).type(dtype), requires_grad=False)

    # define loss function: mean squared error
    loss_function = nn.MSELoss()

    # define optimize method: stochastic gradient descent
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for i in range(iterations):
        # sentence and target are matrix
        for train_sentence, train_target in train_input, train_output:
            
            # zero gradient buffers
            optimizer.zero_grad()

            # clear hidden
            model.hidden = model.init_hidden()
            
            # find output of network
            train_output = lstm_net(train_sentence)

            # error of output and target
            loss = loss_function(train_output, train_target)

            # backpropagate the error
            loss.backward()

            # update the weights
            optimizer.step()

    return lstm_net


def test(data, net):
    """
        Find loss of test set.
    """
    # make data ready for use
    test_input = Variable(torch.FloatTensor(test_data[:,:3]).type(dtype), requires_grad=False)
    test_target = Variable(torch.FloatTensor(test_data[:,-1]).type(dtype), requires_grad=False)
    
    # calculate output
    test_output = net(test_input)
    
    # define loss function: mean squared error
    loss_function = nn.MSELoss()
    
    # calculate the error
    test_loss = loss_function(test_output, test_target)
    
    return test_loss.data[0]


def create_nn(data, iterations, layer_info, k=None, lr=1e-17):  
    """
    Create a neural netwerk.
    Input:
        :parsm data: np array
        :param iterations: number of iterations (non-zero integer)
        :param k: number of folds (non-zero integer)
        :param layer_info: list of integers where element is the number of nodes of corresponding layer
        :param lr: learning rate (default is 1e-17)
    Output: neural netwerk
    """
    
    # not using k-fold cross validation
    if k == None or k <= 1:
        best_net = train(data, lr, iterations, layer_info)
        
    # using k-fold cross validation
    else:
        folds = create_folds(data, k)
        best_net = None
        error = None
        folds_prime = random.sample(folds, 2)
        for fold in folds_prime:
            net = train(fold[0], lr, iterations, layer_info)
            test_loss = test(fold[1], net)
            if error == None or error > test_loss:
                error = test_loss
                best_net = net

    return best_net