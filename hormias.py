import numpy as np
import random
import matplotlib.pyplot as plt

class ACO_TSP:
    def __init__(self, distancias, alpha=1, beta=2, rho=0.1, Q=1, n_hormigas=10):
        """
        distancias: matriz de distancias entre ciudades
        n_hormigas: número de hormigas
        alpha: importancia de las feromonas
        beta: importancia de la visibilidad (1/distancia)
        rho: tasa de evaporación
        Q: constante para depositar feromonas
        """
        self.distancias = distancias
        self.n = len(distancias)
        self.n_hormigas = n_hormigas
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Inicializar feromonas
        self.feromonas = np.ones((self.n, self.n))
        
        # Calcular visibilidad (inversa de la distancia)
        self.visibilidad = 1 / (distancias + 1e-10)  # Evitar división por cero
        
    def run(self, iteraciones):
        mejor_ruta = None
        mejor_distancia = float('inf')
        historial_distancias = []
        
        for it in range(iteraciones):
            rutas = self._construir_rutas()
            self._actualizar_feromonas(rutas)
            
            # Encontrar la mejor ruta de esta iteración
            for ruta, distancia in rutas:
                if distancia < mejor_distancia:
                    mejor_distancia = distancia
                    mejor_ruta = ruta
            
            historial_distancias.append(mejor_distancia)
            
            if (it + 1) % 10 == 0:
                print(f"Iteración {it+1}: Mejor distancia = {mejor_distancia:.2f} km")
        
        return mejor_ruta, mejor_distancia, historial_distancias
    
    def _construir_rutas(self):
        rutas = []
        for _ in range(self.n_hormigas):
            ruta, distancia = self._construir_ruta()
            rutas.append((ruta, distancia))
        return rutas
    
    def _construir_ruta(self):
        ruta = []
        distancia_total = 0
        
        # Empezar en una ciudad aleatoria
        ciudad_actual = random.randint(0, self.n-1)
        ruta.append(ciudad_actual)
        ciudades_visitadas = set([ciudad_actual])
        
        # Visitar todas las ciudades
        while len(ciudades_visitadas) < self.n:
            # Calcular probabilidades para la siguiente ciudad
            probabilidades = []
            for ciudad in range(self.n):
                if ciudad not in ciudades_visitadas:
                    feromona = self.feromonas[ciudad_actual][ciudad] ** self.alpha
                    visibilidad = self.visibilidad[ciudad_actual][ciudad] ** self.beta
                    probabilidades.append((ciudad, feromona * visibilidad))
                else:
                    probabilidades.append((ciudad, 0))
            
            # Normalizar probabilidades
            total = sum(p for c, p in probabilidades)
            if total == 0:
                # Si todas las probabilidades son 0, elegir aleatoriamente
                ciudad_siguiente = random.choice([c for c in range(self.n) if c not in ciudades_visitadas])
            else:
                probabilidades = [(c, p/total) for c, p in probabilidades]
                ciudades, probs = zip(*probabilidades)
                ciudad_siguiente = random.choices(ciudades, weights=probs)[0]
            
            # Mover a la siguiente ciudad
            distancia_total += self.distancias[ciudad_actual][ciudad_siguiente]
            ruta.append(ciudad_siguiente)
            ciudades_visitadas.add(ciudad_siguiente)
            ciudad_actual = ciudad_siguiente
        
        # Volver a la ciudad inicial para completar el ciclo
        distancia_total += self.distancias[ruta[-1]][ruta[0]]
        ruta.append(ruta[0])
        
        return ruta, distancia_total
    
    def _actualizar_feromonas(self, rutas):
        # Evaporación
        self.feromonas *= (1 - self.rho)
        
        # Depositar feromonas
        for ruta, distancia in rutas:
            contribucion = self.Q / distancia
            for i in range(len(ruta)-1):
                ciudad_actual = ruta[i]
                ciudad_siguiente = ruta[i+1]
                self.feromonas[ciudad_actual][ciudad_siguiente] += contribucion
                self.feromonas[ciudad_siguiente][ciudad_actual] += contribucion  # Matriz simétrica

# Ejemplo de uso con ciudades del Perú - Región Puno y alrededores
if __name__ == "__main__":
    # Ciudades de la región de Puno y importantes ciudades cercanas del sur del Perú
    ciudades = ["Puno", "Juliaca", "Ilave", "Juli", "Lampa", "Ayaviri", "Desaguadero"]
    
    # Matriz de distancias aproximadas en kilómetros entre estas ciudades
    # Basadas en distancias reales por carretera
    distancias = np.array([
        #   Puno  Jul   Ila   Jul   Lam   Aya   Des
        [   0,   44,   54,   80,   82,  126,  148],  # Puno
        [  44,    0,   98,  124,   32,   84,  192],  # Juliaca
        [  54,   98,    0,   26,  136,  180,   94],  # Ilave
        [  80,  124,   26,    0,  162,  206,  120],  # Juli
        [  82,   32,  136,  162,    0,   52,  224],  # Lampa
        [ 126,   84,  180,  206,   52,    0,  276],  # Ayaviri
        [ 148,  192,   94,  120,  224,  276,    0]   # Desaguadero
    ])
    
    print("=== ALGORITMO DE COLONIA DE HORMIGAS PARA TSP ===")
    print("Ciudades de la región de Puno, Perú")
    print("Ciudades incluidas:", ", ".join(ciudades))
    print("\nMatriz de distancias (km):")
    print("      ", end="")
    for ciudad in ciudades:
        print(f"{ciudad[:4]:>6}", end="")
    print()
    for i, ciudad in enumerate(ciudades):
        print(f"{ciudad[:8]:<8}", end="")
        for j in range(len(ciudades)):
            print(f"{distancias[i][j]:>6.0f}", end="")
        print()
    
    print("\n" + "="*50)
    print("EJECUTANDO ALGORITMO ACO...")
    print("="*50)
    
    # Crear y ejecutar el algoritmo con parámetros optimizados para este problema
    aco = ACO_TSP(distancias, 
                  n_hormigas=20,     # Más hormigas para mejor exploración
                  alpha=1,           # Importancia de feromonas
                  beta=3,            # Mayor importancia a la distancia
                  rho=0.15,          # Tasa de evaporación moderada
                  Q=100)             # Constante para depositar feromonas
    
    mejor_ruta, mejor_distancia, historial = aco.run(iteraciones=100)
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("RESULTADOS FINALES")
    print("="*50)
    print("\nMejor ruta encontrada:")
    ruta_nombres = [ciudades[i] for i in mejor_ruta]
    print(" → ".join(ruta_nombres))
    print(f"\nDistancia total del recorrido: {mejor_distancia:.1f} km")
    
    # Mostrar distancias entre ciudades consecutivas en la mejor ruta
    print("\nDetalle del recorrido:")
    distancia_acumulada = 0
    for i in range(len(mejor_ruta)-1):
        ciudad_actual = mejor_ruta[i]
        ciudad_siguiente = mejor_ruta[i+1]
        distancia_tramo = distancias[ciudad_actual][ciudad_siguiente]
        distancia_acumulada += distancia_tramo
        print(f"{ciudades[ciudad_actual]} → {ciudades[ciudad_siguiente]}: {distancia_tramo} km (Acumulado: {distancia_acumulada:.1f} km)")
    
    # Gráfico de convergencia
    plt.figure(figsize=(12, 5))
    
    # Gráfico de convergencia
    plt.subplot(1, 2, 1)
    plt.plot(historial, 'b-', linewidth=2)
    plt.title("Convergencia del Algoritmo ACO\nCiudades de Puno, Perú")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor distancia (km)")
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    
    # Gráfico de barras con distancias entre ciudades principales
    plt.subplot(1, 2, 2)
    algunas_distancias = [
        ("Puno-Juliaca", distancias[0][1]),
        ("Juliaca-Lampa", distancias[1][4]),
        ("Puno-Juli", distancias[0][3]),
        ("Ilave-Juli", distancias[2][3]),
        ("Lampa-Ayaviri", distancias[4][5])
    ]
    nombres, vals = zip(*algunas_distancias)
    plt.bar(range(len(nombres)), vals, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
    plt.title("Distancias entre Ciudades Principales")
    plt.ylabel("Distancia (km)")
    plt.xticks(range(len(nombres)), nombres, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas adicionales
    print(f"\nEstadísticas del algoritmo:")
    print(f"- Número de iteraciones: 100")
    print(f"- Número de hormigas por iteración: 20")
    print(f"- Mejora total: {historial[0]:.1f} → {historial[-1]:.1f} km ({((historial[0]-historial[-1])/historial[0]*100):.1f}% de mejora)")
    print(f"- Distancia promedio entre ciudades: {np.mean(distancias[distancias > 0]):.1f} km")