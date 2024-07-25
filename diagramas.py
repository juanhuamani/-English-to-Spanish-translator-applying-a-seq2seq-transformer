import pickle
import matplotlib.pyplot as plt

# Cargar el historial de entrenamiento
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Graficar la pérdida
plt.plot(history['loss'], label='Pérdida de entrenamiento')
plt.plot(history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Curva de Pérdida')
plt.show()

# Graficar la exactitud
plt.plot(history.get('accuracy', []), label='Exactitud de entrenamiento')
plt.plot(history.get('val_accuracy', []), label='Exactitud de validación')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()
plt.title('Curva de Exactitud')
plt.show()
