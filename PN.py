import random

def gaussNormalize(x):
    xMean = np.tile(x.mean(axis=0), (len(x),1))
    xStd = np.tile(x.std(axis=0), (len(x),1))
    normalizedX = (x - xMean) / xStd;
    return normalizedX
    
def minMaxNormalize(x):
    xMin = np.tile(x.min(axis=0), (len(x),1))
    xMax = np.tile(x.max(axis=0), (len(x),1))
    normalizedX = (x - xMin) / (xMax-xMin)
    return normalizedX

path = '/Users/tmold_000/Documents/Idan' # the path for the files


# This chunk of code is to load the limits of each conductance 
scaleF = []
for l in open(path +'/paramconfig','r'):
    if 'Max' in l:
        tm = [l.split()[1]]
    if 'Min' in l:
        tm.append(l.split()[1])
        scaleF.append(array(tm))

scaleF = array(scaleF)
minV = scaleF[:,0].astype(float) # an array of the minimum values
maxV = scaleF[:,1].astype(float) # an array of the maximum values


# Here I load the data while normalize with the max and min
X = []
Y = []
for l in open(path + '/whole_population.txt','r'):
    if 'Generation' not in l and '\n' !=l:
        if max(array(l.split()[1:5]).astype(float32))<5:
            Y.append(array(l.split()[1:5]).astype(float32)/5)
            X.append(abs((array(l.split()[63:]).astype(float32)))) # Not sure if this line is correct.



X1 = array(X)#[len(X)/2:])
y1 = array(Y)#[len(Y)/2:])



inds = random.sample(xrange(len(X1)),30000) # now I take 30,000 random samples

x = gaussNormalize(array([X1[i] for i in inds]).astype(float32))
y = gaussNormalize(array([y1[i] for i in inds]).astype(float32))

indstest = random.sample(xrange(len(X1)),30000) # another 30,000 random samples for the test
xtest = gaussNormalize(array([X1[i] for i in indstest]).astype(float32))
ytest = gaussNormalize(array([y1[i] for i in indstest]).astype(float32))





# This is where is code starts, the NN code starts
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

net1 = NeuralNet(
    layers=[  # four layers: two hidden layers
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 31),  # 31 input
    hidden_num_units=400,  # number of units in hidden layer
    hidden1_num_units=400,
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=4,  # 4 outputs

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=450,  # we want to train this many epochs
    verbose=1,
    )



net1.fit(x, y) # This thing try to do the fit itself



y_pred = net1.predict(xtest)  # Here I test the other 30k random samples
error = ytest-y_pred
originalErrors = sum(ytest*5, axis = 1)
distances = sum(error*5, axis = 1)
meanOriginalErrors = mean(originalErrors)
meanDistances = mean(distances)
print(meanOriginalErrors) 
print(meanDistances)




