import os
from sacred import Experiment
from sacred.observers import MongoObserver


def create_experiment():
    ex = Experiment()
    # Set up database logs
    uri = os.environ.get('MLAB_URI')
    database = os.environ.get('MLAB_DB')
    if all([uri, database]):
        ex.observers.append(MongoObserver(uri, database))
    else:
        print('Running without Sacred observers')

    return ex
