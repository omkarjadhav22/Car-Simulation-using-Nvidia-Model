print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from utilis import importDataInfo, balanceData, loadData, createModel, batchGen
from sklearn.model_selection import train_test_split  # Importing train_test_split from sklearn.model_selection
import matplotlib.pyplot as plt

path = 'mydata'
data = importDataInfo(path)

data = balanceData(data, display=False)
imagesPath, steerings = loadData(path, data)

# Splitting data into training and validation sets
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)

print('Total training images:', len(xTrain))
print('Total validation images:', len(xVal))

model = createModel()
model.summary()

history = model.fit(batchGen(xTrain, yTrain, 100, 1),
                                  steps_per_epoch=100,
                                  epochs=10,
                                  validation_data=batchGen(xVal, yVal, 100, 0),
                                  validation_steps=200)

model.save('model.h5')
print('Model Saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
