import tensorflow.keras as keras
import numpy as np
MAX_LENGTH = 34
NORMALIZE = 255
model = keras.models.load_model('models/autocompleteNet.h5')
def get_content(file):
    with open(file) as glosario:
        content = glosario.read()
        content = content.split('\n')
        content = content[:-1]
    return content

def standardize(array_word):
    array_ascii = []
    for letter in array_word:
        #ord() convierte a ASCII
        array_ascii.append(round(ord(letter)/NORMALIZE,8))
    #estandarizar a una unica longitud las palabras
    if len(array_ascii) < MAX_LENGTH:
        zeros = list(np.zeros(MAX_LENGTH - len(array_ascii)))
        array_ascii = array_ascii + zeros
    return np.expand_dims(array_ascii,axis=0)
output = get_content('dataset/glosario.txt')

while True:
    word = str(input())
    pred = standardize(word)
    predictions = model.predict(pred)
    for i in range(0,3):
        p = np.argmax(predictions)
        predictions = np.delete(predictions , int(p))
        print(output[p])
