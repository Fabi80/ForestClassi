import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from tensorflow.keras import Model
from keras import Sequential,layers,optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class data_prepro():
    #will not be instantiated - for clean code purpose only
    @classmethod
    def load_csv(cls):
        data = pd.read_csv('cover_data.csv')
      #  print(data.describe())
        return data

    @classmethod
    def get_labels(cls,data):
        labels = data.iloc[:,54]
        return labels
    @classmethod
    def get_feature(cls,data):
        features = data.iloc[:,:54]
        print(features.describe())
        return features

    #scale features down with sklearnStandardScaler
    @classmethod
    def scale_features(cls,features):
        #scale nur die Features, die größer als 1 werden können
        ct = ColumnTransformer([("scale",StandardScaler(),['Elevation','Aspect','Horizontal_Distance_To_Hydrology',
                                                           "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
                                                           "Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"])],remainder='passthrough')
        scaled_features = pd.DataFrame(ct.fit_transform(features))
      #  print(scaled_features.describe())
        return scaled_features

    # convert integers to dummy variables
    @classmethod
    def convert_labels_categorical(self,labels):
        dummy_y = to_categorical(labels)
        return dummy_y

    @classmethod
    def split_train_validation(cls,features,labels):
        features_train,features_test,labels_train,labels_test = train_test_split(features,labels,train_size=0.8,random_state=42)
        return features_train,features_test,labels_train,labels_test


class Classi_model():
    def __init__(self,features_train,features_test,labels_train,labels_test):
        self.features_train = features_train
        self.features_test = features_test
        self.labels_train = labels_train
        self.labels_test = labels_test

    def build_model(self):
        self.my_model = Sequential()
        self.my_model.add(layers.InputLayer(input_shape=(self.features_train.shape[1],)))
        self.my_model.add(layers.Dense(53, activation='relu'))
        self.my_model.add(layers.Dropout(0.1))
        self.my_model.add(layers.Dense(27, activation='relu'))
        self.my_model.add(layers.Dropout(0.1))
     #   self.my_model.add(layers.Dense(53, activation='relu'))
        self.my_model.add(layers.Dense(8, activation='softmax'))
        self.my_model.summary()
        self.my_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

    def train_model(self):
        # early stopping implementation
        es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=4)
        #trainthemodel
        history = self.my_model.fit(self.features_train, self.labels_train,validation_data=(self.features_test,self.labels_test),epochs=18,batch_size=53,callbacks=es,verbose=1)
        loss,acc = self.my_model.evaluate(self.features_test,self.labels_test)
        print('-------here comes the accuracy:')
        print('log:',loss,'  acc:',acc)

        #use sklearnperformance metrics
        y_estimate = self.my_model.predict(self.features_test, verbose=1)
        y_estimate = np.argmax(y_estimate, axis=1)
        y_true = np.argmax(self.labels_test, axis=1)
        class_names = ['Spruce/Fir', 'Lodgepole Pine',
                       'Ponderosa Pine', 'Cottonwood/Willow',
                       'Aspen', 'Douglas-fir', 'Krummholz']
        print(classification_report(y_true, y_estimate,target_names=class_names))
        return history

    def save_model(self):
        print("----saving model--------")
        self.my_model.save('bfmodel')


class Plot():
    @classmethod
    def plot(self,history):
        print("----here comes the plot----")
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('model accuracy')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('accuracy')
        ax1.legend(['training', 'validation'], loc='upper left')


        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('model loss')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
        ax2.legend(['train', 'validation'], loc='upper left')
        plt.show()
        print("---end of plot---")




data = data_prepro.load_csv()
labels = data_prepro.get_labels(data)
categorical_labels = data_prepro.convert_labels_categorical(labels)
features = data_prepro.get_feature(data)
scaled_features = data_prepro.scale_features(features)
features_train,features_test,labels_train,labels_test = data_prepro.split_train_validation(scaled_features,categorical_labels)
print(features_train.describe())
my_model = Classi_model(features_train,features_test,labels_train,labels_test)
my_model.build_model()
history = my_model.train_model()
Plot.plot(history)
#my_model.save_model()

