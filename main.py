from keras.optimizers import SGD
from keras import *
from keras.layers import *
from keras.models import *
from convnetskeras.convnets import preprocess_image_batch, convnet
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import pickle


def make_extractor_model():
    model = convnet('alexnet',weights_path="alexnet_weights.h5", heatmap=False)
    model.layers = model.layers[:-4] 
    model = Model(input=model.input,output=model.layers[-1].output)
    return model

def make_model(hidden_layer_neurons):
    REMOVAL_LAYERS = 4
    SECOND_OUTPUT_DIM = 1

    return Sequential([
        Dense(hidden_layer_neurons, input_dim=4096,name="cust_dense_1",init="uniform"),
        Activation('tanh'),
        Dense(1),
        Activation('tanh')])



def flatten_once(l):
    return [item for sublist in l for item in sublist]

def split_train_validate(l,fraction):
    shuffled_data = sorted(l,key=lambda k: random.random())
    return (shuffled_data[int(fraction * len(shuffled_data)):], shuffled_data[:int(fraction * len(shuffled_data))])

# Plot data
def plot_results(y_test, y_score, title,location,show=False):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    if roc_auc < .7:
        return #Fuck it
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    full_title = 'ROC,' + ('AUC: %0.2f' % roc_auc) +  str(title) 
    plt.title(full_title)
    try:
        os.mkdir("./roc_curves/" + location)
    except Exception:
        #It prolly exists
        pass
    plt.savefig("./roc_curves/" +location+ "/" + full_title +".png".replace("\n","").replace(" ", "_"))
    if(show):
        plt.show()
    plt.close()


class CustTrainingResults:
    def __init__(self,test_targets,test_outputs,validation_targets,validation_outputs,training_targets,training_outputs):
        self.test_targets = test_targets
        self.test_outputs = test_outputs
        self.validation_targets = validation_targets
        self.validation_outputs = validation_outputs
        self.training_targets = training_targets
        self.training_outputs = training_outputs

def train_all_images(model,epochs,validation_split_fraction):
    BATCH_SIZE = 100
    training_imgs = []
    training_targets = []
    testing_imgs = []
    testing_targets = []
    for root, dirs, files in os.walk("./LiveDet11AlexnetExtracted"):
        for f in files:
            fname = os.path.join(root,f)
            if "Training" in fname:
                training_imgs.append(fname)
                if "Spoof" in fname:
                    training_targets.append([1])
                else:
                    training_targets.append([0])
            if "Testing" in fname:
                testing_imgs.append(fname)
                if "Spoof" in fname:
                    testing_targets.append([1])
                else:
                    testing_targets.append([0])
    fraction_using = 1
    training = list(zip(training_imgs,training_targets))
    testing = list(zip(testing_imgs,testing_targets))
    training = random.sample(training,int(len(training) / fraction_using))
    testing = random.sample(testing,int(len(testing) / fraction_using))
    training,validation = split_train_validate(training,validation_split_fraction)
    
    cur_images = []
    for f in map(lambda a:a[0],training):
        with open(f, 'rb') as f:
            cur_images.append(pickle.load(f))
    cur_images = np.array(cur_images)
    cur_targets = list(map(lambda a:a[1], training))
    print(cur_images)

    validation_images = []
    for f in map(lambda a:a[0],validation):
        with open(f, 'rb') as f:
            validation_images.append(pickle.load(f))
    validation_images = np.array(validation_images)
    validation_targets = list(map(lambda a:a[1], validation))


    model.fit(cur_images, cur_targets, nb_epoch=100,shuffle=True,validation_data=(validation_images,validation_targets),batch_size=BATCH_SIZE)

    testing_imgs = []
    for f in map(lambda a:a[0],testing):
        with open(f, 'rb') as f:
            testing_imgs.append(pickle.load(f))
    testing_imgs = np.array(testing_imgs)

    print(testing_imgs) 
    return  CustTrainingResults(list(map(lambda a: a[1], testing)),model.predict(testing_imgs,batch_size=BATCH_SIZE,verbose=1),\
                list(map(lambda a: a[1], validation)),model.predict(validation_images,batch_size=BATCH_SIZE,verbose=1), \
                list(map(lambda a: a[1], training)),model.predict(cur_images,batch_size=BATCH_SIZE,verbose=1))



    return (list(map(lambda a: a[1], testing)),model.predict(testing_imgs,batch_size=BATCH_SIZE,verbose=1))


def preprocess_alexnet(imgs):
    return preprocess_image_batch(imgs,img_size=(256,256), crop_size=(227,227))


def extract_features(model):
    images = []
    for root, dirs, files in os.walk("."):
        for f in files:
            fname = os.path.join(root,f)
            if "Digital" in fname or "Sagem" in fname:
                images.append(fname)
    features = model.predict(preprocess_alexnet(images),verbose=1)
    for fname,feature in zip(images,features):
        with open("./LiveDet11AlexnetExtracted" + (fname[10:].replace("/","")),"wb") as f:
                pickle.dump(feature,f)


def train_and_plot(model,epochs,validation_split_fraction,title):
    results =  train_all_images(model,epochs,validation_split_fraction)
    plot_results(results.training_targets,results.training_outputs,  "Training",title)
    plot_results(results.validation_targets,results.validation_outputs, "Validation",title)
    plot_results(results.test_targets,results.test_outputs, "Testing",title)

EPOCHS = 100
VALIDATION_SPLIT = 1/10

def test_sgd(learning_rate,momentum,decay, hidden_layer_neurons):
    model = make_model(hidden_layer_neurons)
    sgd = SGD(lr=learning_rate, momentum=momentum,decay=decay)
    model.compile(optimizer=sgd, loss="mse")
    title = "SGD Optimizer, Hidden Layers: " + str(hidden_layer_neurons) +"\n Learning Rate: " + str(learning_rate) + ", Momentum: " + str(momentum) + ", Learning Rate Decay: " + str(decay)
    train_and_plot(model,EPOCHS,VALIDATION_SPLIT,title)

#test_sgd(0.01,1e-2,0,50)
#test_sgd(0.1,1e-2,0,50)
#test_sgd(0.001,1e-2,0,50)
#test_sgd(0.0001,1e-2,0,50)
#test_sgd(0.00001,1e-2,0,50)






class CustLeaveOutResults:
    def __init__(self,left_out_material,training_targets,training_outputs,left_out_targets,left_out_output):
        self.left_out_material = left_out_material
        self.training_targets = training_targets
        self.training_outputs = training_outputs
        self.left_out_targets = left_out_targets
        self.left_out_output = left_out_output

#Begin Leave out tests
def train_leave_one_out(model,out_type):
    BATCH_SIZE = 100
    training_imgs = []
    training_targets = []
    testing_imgs = []
    testing_targets = []
    for root, dirs, files in os.walk("./LiveDet11AlexnetExtracted"):
        for f in files:
            fname = os.path.join(root,f)
            if out_type in fname:
                testing_imgs.append(fname)
                if "Spoof" in fname:
                    testing_targets.append([1])
                else:
                    testing_targets.append([0])
            else:
                training_imgs.append(fname)
                if "Spoof" in fname:
                    training_targets.append([1])
                else:
                    training_targets.append([0])
    
    training = list(zip(training_imgs,training_targets))
    testing = list(zip(testing_imgs,testing_targets))

    training, extra_testing_samples = split_train_validate(training,1/10) 
    #Only include non spoof 
    testing += list(filter(lambda a: a[1] == [0], extra_testing_samples))


    cur_images = []
    for f in map(lambda a:a[0],training):
        with open(f, 'rb') as f:
            cur_images.append(pickle.load(f))
    cur_images = np.array(cur_images)
    cur_targets = list(map(lambda a:a[1], training))

    model.fit(cur_images, cur_targets, nb_epoch=100,shuffle=True,batch_size=BATCH_SIZE)

    testing_imgs = []
    for f in map(lambda a:a[0],testing):
        with open(f, 'rb') as f:
            testing_imgs.append(pickle.load(f))
    testing_imgs = np.array(testing_imgs)
    return CustLeaveOutResults(out_type,
            list(map(lambda a: a[1], training)),model.predict(cur_images,batch_size=BATCH_SIZE,verbose=1),\
            list(map(lambda a: a[1], testing)),model.predict(testing_imgs,batch_size=BATCH_SIZE,verbose=1))

def test_leave_out(learning_rate,momentum,decay, hidden_layer_neurons):
    types = ["Gelatine","Latex","Playdoh","Silicone","WoodGlue"]
    for type in types:
        model = make_model(hidden_layer_neurons)
        sgd = SGD(lr=learning_rate, momentum=momentum,decay=decay)
        model.compile(optimizer=sgd, loss="mse")
        title = "Leave Out " + type
        results = train_leave_one_out(model,type)
        title = " Training without type" + type  + " "
        plot_results(results.training_targets,results.training_outputs,title,type)
        title = " Testing when " + type + " is introduced"
        plot_results(results.left_out_targets,results.left_out_output,title,type)


test_leave_out(.01,.001,0,50)
