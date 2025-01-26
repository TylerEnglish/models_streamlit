import joblib
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

class SimpleTorchModel(nn.Module):
    def __init__(self):
        super(SimpleTorchModel, self).__init__()
        self.fc1 = nn.Linear(5, 10)  
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def create_tf_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),  
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

class ComplexTorchModel(nn.Module):
    def __init__(self):
        super(ComplexTorchModel, self).__init__()
        self.layer1 = nn.Linear(5, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        return self.output(x)

# Complex TensorFlow Model
def create_complex_tf_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(5,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def ensure_model_directory():
    if not os.path.exists("models"):
        os.makedirs("models")

def load_model(model_name):
    ensure_model_directory()
    if model_name == "Complex PyTorch Model":
        model = ComplexTorchModel()
        model.load_state_dict(torch.load("models/complex_pytorch_model.pth"))
        model.eval()
        return model
    elif model_name == "Complex TensorFlow Model":
        return tf.keras.models.load_model("models/complex_tf_model.keras")
    elif model_name == "PyTorch Neural Network":
        model = SimpleTorchModel()
        model.load_state_dict(torch.load("models/pytorch_model.pth"))
        model.eval()
        return model
    elif model_name == "TensorFlow Neural Network":
        return tf.keras.models.load_model("models/tensorflow_model.keras")
    elif model_name == "Linear Regression":
        return joblib.load("models/linear_regression.pkl")
    elif model_name == "Random Forest":
        return joblib.load("models/random_forest.pkl")
    elif model_name == "Neural Network":
        return joblib.load("models/neural_network.pkl")
    else:
        raise FileNotFoundError(f"Model {model_name} not found. Please train the models.")

def make_prediction(model, features):
    features = np.array(features).reshape(1, -1)
    if isinstance(model, nn.Module):
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return model(features_tensor).item()
    elif isinstance(model, tf.keras.Model):
        return model.predict(features)[0][0]
    else:
        return model.predict(features)[0]

def train_models():
    ensure_model_directory()
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)

    # Train Linear Regression
    lr_model = LinearRegression().fit(X, y)
    joblib.dump(lr_model, "models/linear_regression.pkl")

    # Train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100).fit(X, y)
    joblib.dump(rf_model, "models/random_forest.pkl")

    # Train Simple Neural Network (MLP)
    nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500).fit(X, y)
    joblib.dump(nn_model, "models/neural_network.pkl")

    # Train Simple PyTorch Model
    simple_torch_model = SimpleTorchModel()
    optimizer_simple = torch.optim.Adam(simple_torch_model.parameters(), lr=0.01)
    criterion_simple = nn.MSELoss()
    X_tensor_simple = torch.tensor(X, dtype=torch.float32)
    y_tensor_simple = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(500):
        optimizer_simple.zero_grad()
        outputs = simple_torch_model(X_tensor_simple)
        loss = criterion_simple(outputs, y_tensor_simple)
        loss.backward()
        optimizer_simple.step()

    torch.save(simple_torch_model.state_dict(), "models/pytorch_model.pth")

    # Train Simple TensorFlow Model
    tf_model = create_tf_model()
    tf_model.fit(X, y, epochs=100, verbose=0) 
    tf_model.save("models/tensorflow_model.keras")

    # Train Complex PyTorch Model
    complex_torch_model = ComplexTorchModel()
    optimizer_complex = torch.optim.Adam(complex_torch_model.parameters(), lr=0.001)
    criterion_complex = nn.MSELoss()
    X_tensor_complex = torch.tensor(X, dtype=torch.float32)
    y_tensor_complex = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(100):
        optimizer_complex.zero_grad()
        outputs = complex_torch_model(X_tensor_complex)
        loss = criterion_complex(outputs, y_tensor_complex)
        loss.backward()
        optimizer_complex.step()

    torch.save(complex_torch_model.state_dict(), "models/complex_pytorch_model.pth")

    # Train Complex TensorFlow Model
    complex_tf_model = create_complex_tf_model()
    complex_tf_model.fit(X, y, epochs=500, verbose=0)
    complex_tf_model.save("models/complex_tf_model.keras")
