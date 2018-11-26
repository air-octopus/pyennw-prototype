# pyennw-prototype

Основной класс программы -- **Engine** (engine.py).
При создании объекта **Engine**:
* открывается sqlite-база (**NeuralNetworkDB**, ennwdb.py), содержащая все данные программы
* из базы загружаются настройки (**Config**, сonfig.py)
* загружаются входные данные для тренировки сети (**TrainingData**, training-data.py)
* создается загрузчик нейронных сетей из базы данных (**NeuralNetworkLoader**, neural_network_loader.py).
  

