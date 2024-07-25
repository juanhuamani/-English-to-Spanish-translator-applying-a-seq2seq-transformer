import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el historial de entrenamiento
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Extraer datos
epochs = range(1, len(history['accuracy']) + 1)
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

# Crear un DataFrame
data = {
    'Epoch': epochs,
    'Training Accuracy': accuracy,
    'Validation Accuracy': val_accuracy,
    'Training Loss': loss,
    'Validation Loss': val_loss
}
df = pd.DataFrame(data)

# Mostrar la tabla
print(df)

# Guardar el DataFrame en un archivo CSV
df.to_csv('training_validation_comparison.csv', index=False)

# Graficar
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()
