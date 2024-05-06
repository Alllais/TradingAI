import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def check_and_load_model(model_path, input_shape, action_size):
    if os.path.exists(model_path):
        print("Loading existing model...")
        return load_model(model_path)
    else:
        print("No existing model found. Creating a new one...")
        return create_q_model(input_shape, action_size)
    
def create_q_model(input_shape, action_size):
    model = Sequential()
    model.add(Dense(24, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
    return model

def save_model(model, path, filename="model.keras"):
    full_path = os.path.join(path, filename)
    model.save(full_path)
