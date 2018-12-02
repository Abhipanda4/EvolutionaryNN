import tensorflow as tf
import numpy as np
from tqdm import tqdm
from operator import itemgetter

from tensorflow.examples.tutorials.mnist import input_data
try:
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
except:
    print("Working Internet Connection Required!!")
    exit()

INPUT_SIZE = 784
OUTPUT_SIZE = 10

class NeuralNetwork(object):
    def __init__(self, parameters=None):
        '''
        Instantiates the network model.
        @inputs:
            parameters: A dict containing no. of layers, sizes of each layer(a list),
                     activation    functions for each layer(a list again) and optimizer
        '''
        self.nn_parameters = parameters
        self.layer_dims = list(parameters["layer_dims"])
        self.activations = list(parameters["activations"])
        print(self.layer_dims)
        print(self.activations)
        print(self.nn_parameters["optimizer"])
        for l in range(1, self.nn_parameters["nb_layers"]):
            weight = tf.random_normal([self.layer_dims[l - 1], self.layer_dims[l]])
            bias = tf.zeros([1, self.layer_dims[l]])
            activation = self.activations[l]
            self.nn_parameters["W" + str(l)] = tf.Variable(weight)
            self.nn_parameters["b" + str(l)] = tf.Variable(bias)
            self.nn_parameters["activation" + str(l)] = self.activations[l]
        self.fitness = 0

    def get_activation(self, l, A_prev, activation):
        '''
        A function to compute the activation of layer l.
        @input:
            l:             layer number
            A_prev:     activation of previous layer
            activation: a string stating the type of activation to be applied;
                        can be one of the following:
                        "relu", "sigmoid", "elu", "leaky_relu", "tanh", 
        '''
        Zl = tf.add(tf.matmul(A_prev, self.nn_parameters["W" + str(l)]), self.nn_parameters["b" + str(l)])
        if activation == "relu":
            Al = tf.nn.relu(Zl)
        elif activation == "sigmoid":
            Al = tf.nn.sigmoid(Zl)
        elif activation == "elu":
            Al = tf.nn.elu(Zl)
        elif activation == "leaky_relu":
            Al = tf.nn.leaky_relu(Zl)
        elif activation == "tanh":
            Al = tf.nn.tanh(Zl)
        elif activation == "None":
            Al = Zl
        return Al

    def compute_flow(self, X):
        '''
        Implements the forward propagation step
        @input:
            X: training data mini-batch
        '''
        A_prev = X
        for l in range(1, self.nn_parameters["nb_layers"]):
            Al = self.get_activation(l, A_prev, self.nn_parameters["activation" + str(l)])
            A_prev = Al
        return Al

    def get_optimizer(self, loss, optimizer):
        if optimizer == "Adam":
            res = tf.train.AdamOptimizer().minimize(loss)
        elif optimizer == "Adadelta":
            res = tf.train.AdadeltaOptimizer(0.8).minimize(loss)
        elif optimizer == "Adagrad":
            res = tf.train.AdagradOptimizer(0.001).minimize(loss)
        elif optimizer == "Momentum":
            res = tf.train.MomentumOptimizer(0.001, 0.5).minimize(loss)
        elif optimizer == "RMSProp":
            res = tf.train.RMSPropOptimizer(0.001).minimize(loss)
        elif optimizer == "sgd":
            res = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        return res

    def run_model(self, num_epochs=15, batch_size=100):
        # decalre placeholders for input and output
        X = tf.placeholder("float", shape=[None, INPUT_SIZE])
        Y = tf.placeholder("float", shape=[None, OUTPUT_SIZE])

        # forward-pass
        logits = self.compute_flow(X)
        # calculate cross entropy loss, no genetics on loss function for now
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        # decide the best optimizer based on evolution
        optimizer = self.get_optimizer(loss, self.nn_parameters["optimizer"])

        with tf.Session() as sess:
            # initialize all global vars in the graph
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                avg_cost = 0
                num_batches = int(mnist.train.num_examples / batch_size)
                for _ in range(num_batches):
                    # procure next batch of data
                    # special helper function for MNIST
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, loss], feed_dict={
                                        X: batch_x,
                                        Y: batch_y
                                   })
                    avg_cost += c
                avg_cost /= num_batches
                # print(avg_cost)

            pred = tf.nn.softmax(logits)
            correct_predn = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predn, "float"))
            self.score = accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
            print(self.score)
            return self.score


class Genetic(object):
    '''
    Class to implement genetic algorithm on neural network architecture
    '''
    def __init__(self):
        '''
        Instantiates:
            nb_layers: a list having all possible number of hidden layers
                        in our network
            nb_nodes : a list having all possible number of nodes in each
                        hidden layer in our network
        '''
        self.param_choices = ["nb_layers", "nb_nodes", "activations", "optimizers"]
        self.nb_layers = range(3, 7)
        self.nb_nodes = [64, 84, 96, 128, 196, 256, 324, 420, 512, 720, 960, 1024]
        self.optimizers = ["Adam", "Adagrad", "Adadelta", "Momentum", "RMSProp", "sgd"]
        self.activations = ["relu", "sigmoid", "elu", "leaky_relu", "tanh"]

    def create_population(self, strength):
        '''
        Creates a random population in the generation.
        @input:
            strength: number of members in the generation
        '''
        members = []
        for _ in range(strength):
            # create a random neural net architecture
            nb_layers = np.random.choice(self.nb_layers)
            parameters = dict()
            parameters["nb_layers"] = nb_layers
            parameters["layer_dims"] = list()
            parameters["activations"] = list()
            # input layer
            parameters["layer_dims"].append(INPUT_SIZE)
            parameters["activations"].append("None")
            for _ in range(1, nb_layers - 1):
                nb_node = np.random.choice(self.nb_nodes)
                activation = np.random.choice(self.activations)
                parameters["layer_dims"].append(nb_node)
                parameters["activations"].append(activation)
            # output layer
            parameters["layer_dims"].append(OUTPUT_SIZE)
            parameters["activations"].append("None")
            parameters["optimizer"] = np.random.choice(self.optimizers)
            members.append(parameters)
        return members

    def breed(self, mother, father, K=1):
        '''
        create K children network by mixing parameters of mother and father
        @input:
            mother: parameters dictionary of a neural net
            father: parameters dictionary of another neural net
            K: No. of children required
        '''
        children = []
        for _ in range(K):
            child = dict()
            # decide nb_layers first
            m_nb_layers = mother["nb_layers"]
            f_nb_layers = father["nb_layers"]
            child["nb_layers"] = np.random.choice(
                            [m_nb_layers, f_nb_layers]
                        )
            # decide the dominant parent
            nb_layers_max = np.max([m_nb_layers, f_nb_layers])
            if nb_layers_max == m_nb_layers:
                p_dom = mother
            else:
                p_dom = father
            if child["nb_layers"] == nb_layers_max:
                # inherit nodes and activations from the dominant parent
                # TODO: define a dominancy factor and choose according to it
                child["layer_dims"] = p_dom["layer_dims"]
                child["activations"] = p_dom["activations"]
            else:
                # inherit randomly
                inherit = np.random.choice([mother, father])
                child["layer_dims"] = inherit["layer_dims"][:child["nb_layers"]-1]
                inherit = np.random.choice([mother, father])
                child["activations"] = inherit["activations"][:child["nb_layers"]-1]
                # take care of the last output layer
                child["layer_dims"].append(OUTPUT_SIZE)
                child["activations"].append("None")
            child["optimizer"] = np.random.choice(
                                    [mother["optimizer"], father["optimizer"]]
                                 )
            children.append(child)

        return children

    def mutate(self, nn_parameters):
        '''
        randomly mutate parameters of the nn
        @input:
            nn_parameters: dictionary of parameters of the network
        '''
        param_to_be_mutated = np.random.choice(self.param_choices)
        if param_to_be_mutated == "nb_layers":
            new_nb_layers = np.random.choice(self.nb_layers)
            if new_nb_layers > nn_parameters["nb_layers"]:
                # pick random number of nodes for the new layers
                for _ in range(new_nb_layers - nn_parameters["nb_layers"] - 1):
                    nb_node = np.random.choice(self.nb_nodes)
                    nn_parameters["layer_dims"].append(nb_node)
                    activation = np.random.choice(self.activations)
                    nn_parameters["activations"].append(activation)
                # take care of last output layer
                nn_parameters["layer_dims"].append(OUTPUT_SIZE)
                nn_parameters["activations"].append("None")
            else:
                # select a subarray of size `new_nb_layers` for each
                # parameter from the beginning
                nn_parameters["layer_dims"] = nn_parameters["layer_dims"][:new_nb_layers]
                nn_parameters["activations"] = nn_parameters["activations"][:new_nb_layers]
            nn_parameters["nb_layers"] = new_nb_layers
        elif param_to_be_mutated == "optimizer":
            nn_parameters["optimizer"] = np.random.choice(self.optimizers)
        else:
            # choose a random index and modify its nb_node or activation
            index = int(np.random.random() * nn_parameters["nb_layers"])
            if param_to_be_mutated == "activations":
                new_attr = np.random.choice(self.activations)
            else:
                new_attr = np.random.choice(self.nb_nodes)
            nn_parameters[param_to_be_mutated][index] = new_attr

        return nn_parameters

    def evolve(self, population, retain=0.25, select=0.15, mutation=0.15):
        '''
        Train all members of the population and evolve based on their fitness
        This function mutates and breeds new babies.
        @input:
            population: a list of parameter dictionaries of all members
        '''
        # get fitness scores for all members
        trained_pop = [(self.evaluate_fitness(member), member)
                    for member in population]
        # arrange members according to their fitness scores
        sorted_scores = sorted(trained_pop, key=itemgetter(0))
        fittest_members = [pair[1] for pair in sorted_scores]
        accuracy_scores = [pair[0] for pair in sorted_scores]

        #retain some members for next generation
        retain_length = int(retain * len(fittest_members))
        parents = fittest_members[:retain_length]
        # randomly add some low performing networks
        for member in fittest_members[retain_length:]:
            if np.random.random() < select:
                parents.append(member)
        # time to make babies
        # for now, preserve the strength of population
        nb_babies = len(population) - len(parents)
        children = []
        while (len(children) < nb_babies):
            # decide number of babies for a pair of parents randomly
            # TODO: Give more babies to fitter parents
            K = 1 + int(3 * np.random.random())
            # get random mom & dad(not both identical)
            mother = np.random.choice(parents)
            father = None
            while not father == mother:
                father = np.random.choice(parents)
            # now breed
            babies = self.breed(mother, father, K)
            for baby in babies:
                # randomly mutate a baby
                if np.random.random() < mutation:
                    baby = self.mutate(baby)
                if len(children) < nb_babies:
                    children.append(baby)
        # we have evolved
        parents.extend(children)
        return accuracy_scores, parents

    def evaluate_fitness(self, nn_parameters=None):
        nn = NeuralNetwork(nn_parameters)
        accuracy = nn.run_model(num_epochs=25, batch_size=100)
        # accuracy is the fitness score of network
        return accuracy * 100


def evolutionary_NN(nb_gens):
    genome = Genetic()
    population = genome.create_population(strength=40)
    max_score = []
    min_score = []
    avg_score = []
    for _ in range(nb_gens):
        accuracy_scores, new_population = \
            genome.evolve(population, retain=0.4, select=0.2, mutation=0.2)
        # visualize scores of each generation
        max_score.append(accuracy_scores[0])
        min_score.append(accuracy_scores[-1])
        avg_score.append(np.mean(accuracy_scores))
        population = new_population
    # the final population is (probably) the fittest population
    trained_pop = [(self.evaluate_fitness(member), member)
                    for member in population]
    # arrange members according to their fitness scores
    sorted_scores = sorted(trained_pop, key=itemgetter(0))
    accuracy_scores = [pair[0] for pair in sorted_scores]
    # complete the scores for all generations
    max_score.append(accuracy_scores[0])
    min_score.append(accuracy_scores[-1])
    avg_score.append(np.mean(accuracy_scores))
    # select the best member
    fittest_member = sorted_scores[0][1]
    # and print its parameters
    for param in genome.param_choices:
        print(param, ": ", fittest_member[param])
    print("Best Accuracy: ", accuracy_scores[0])

if __name__ == "__main__":
    np.random.seed(0)
    evolutionary_NN(1)
    # dummy = dict()
    # dummy["nb_layers"] = 6
    # dummy["layer_dims"] = [784, 128, 128, 196, 512, 10]
    # dummy["activations"] = ["None", "sigmoid", "elu", "relu", "tanh", "None"]
    # dummy["optimizer"] = "Adadelta"
    # NeuralNetwork(dummy).run_model()
