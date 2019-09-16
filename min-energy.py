from keras.models import load_model
from json import dumps

class MinEnergy:
    def getData(self,response):
        data = dumps(response)
        model = load_model('minEnergyModel.h5')

        model.predict(data)