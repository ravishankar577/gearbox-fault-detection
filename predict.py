from tensorflow import keras
import pandas as pd

model_1 = keras.models.load_model('model')
print("Loaded model from disk")

fault_labels = {
    0 :'healthy',
    1 : 'faulty'}


a1 = 3.9744758713314887
a2 = 4.688735184356668
a3 = 4.446622266788457
a4 = 5.349605018186907 
load = 0.6

input_dict = {"a1": [a1], "a2": [a2], "a3":[a3], "a4":[a4], "load": [load]}

input_df = pd.DataFrame.from_dict(input_dict)

y_pred= model_1.predict(input_df)
prediction = y_pred[0][0]

if prediction >0.5: 
    prediction_class = 1
else:
    prediction_class = 0
prediction_label = fault_labels[prediction_class]

print(prediction_label)
