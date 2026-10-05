import numpy as np

# Datos
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

escala = np.array([100, 50])
X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def entrenar(tasa_aprendizaje, epocas, verbose=False):
    rng = np.random.default_rng(42)  # semilla fija = resultados reproducibles
    pesos = rng.normal(size=(2, 1))
    sesgo = 0.0

    for epoca in range(epocas):
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)
        error = probabilidad - y

        # Gradiente del ECM a través de la sigmoide
        gradiente_z = error * probabilidad * (1 - probabilidad)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if verbose and epoca % (epocas // 10) == 0:
            print(f"Época {epoca:6d} | ECM: {np.mean(error ** 2):.6f}")

    probabilidad = sigmoide(X_normalizado @ pesos + sesgo)
    ecm = float(np.mean((probabilidad - y) ** 2))
    return pesos, sesgo, ecm, probabilidad


def predecir(humedad, temperatura, pesos, sesgo, umbral=0.5):
    nuevo = np.array([humedad, temperatura], dtype=float) / escala  # MISMA escala
    prob = float(sigmoide(nuevo @ pesos + sesgo).item())
    return prob, int(prob >= umbral)


# Entrenamiento base
print("\n=== Entrenamiento base (0.5, 10000) ===")
pesos, sesgo, ecm, prob_train = entrenar(0.5, 10000, verbose=True)
print(f"\nPeso humedad: {pesos[0, 0]:.4f}")
print(f"Peso temperatura: {pesos[1, 0]:.4f}")
print(f"Sesgo: {sesgo:.4f}")
print(f"Error final (ECM): {ecm:.6f}")
print("\nCaso | Humedad | Temp | Prob | Salida | Esperada")
for i in range(len(X)):
    print(f"{i+1:4d} | {X[i,0]:6.0f}% | {X[i,1]:3.0f}° | {prob_train[i,0]:.4f} | "
          f"{int(prob_train[i,0] >= 0.5)} | {int(y[i,0])}")

# Condiciones nuevas
print("\n=== Condiciones nuevas ===")
nuevas = [(75, 30), (45, 34), (25, 22), (50, 25), (30, 40)]
for h, t in nuevas:
    p, d = predecir(h, t, pesos, sesgo)
    print(f"Humedad {h}% | Temp {t}°C | Prob {p:.4f} | Decisión {d}")

# Experimentos
print("\n=== Experimentos ===")
experimentos = [
    ("Base", 0.5, 10000),
    ("Pocas épocas", 0.5, 100),
    ("Intermedia", 0.5, 1000),
    ("Más épocas", 0.5, 20000),
    ("Tasa pequeña", 0.01, 10000),
    ("Tasa moderada", 0.1, 10000),
    ("Tasa alta", 1.0, 10000),
    ("Tasa muy alta", 2.0, 10000),
]
print("Prueba | Tasa | Épocas | ECM | Correctas | Prob(45%,34°C)")
for nombre, tasa, ep in experimentos:
    w, b, e, p = entrenar(tasa, ep)
    correctas = int(np.sum((p >= 0.5).astype(int) == y))
    prob45, _ = predecir(45, 34, w, b)
    print(f"{nombre} | {tasa} | {ep} | {e:.6f} | {correctas}/10 | {prob45:.4f}")

# Umbrales (sin reentrenar)
print("\n=== Umbrales (modelo base) ===")
for umbral in (0.4, 0.5, 0.6):
    train = (prob_train >= umbral).astype(int).flatten().tolist()
    nuevas_u = [predecir(h, t, pesos, sesgo, umbral)[1] for h, t in nuevas]
    print(f"Umbral {umbral}: entrenamiento={train} | nuevas={nuevas_u}")