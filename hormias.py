import numpy as np
import random
import matplotlib.pyplot as plt
from time import time
import pandas as pd

class ACO_TSP_Optimizado:
    def __init__(self, distancias, ciudades, alpha=1, beta=3, rho=0.15, Q=100, n_hormigas=20):
        """
        Versión optimizada basada en el estudio de Puno
        
        Parámetros optimizados según el informe:
        - alpha = 1 (importancia feromonas)
        - beta = 3 (mayor peso a distancia)
        - rho = 0.15 (evaporación moderada)
        - Q = 100 (escala apropiada para distancias)
        - n_hormigas = 20 (balance exploración-explotación)
        """
        self.distancias = distancias
        self.ciudades = ciudades
        self.n = len(distancias)
        self.n_hormigas = n_hormigas
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Inicialización mejorada de feromonas
        self.feromonas = np.ones((self.n, self.n)) * (1 / self.n)
        
        # Visibilidad (inversa de la distancia con manejo de ceros)
        self.visibilidad = 1 / (distancias + np.eye(self.n) * 1e-10)
        
        # Estadísticas
        self.historial = {
            'mejor_distancia': [],
            'promedio_distancias': [],
            'peor_distancia': [],
            'tiempo_iteracion': []
        }
    
    def run(self, iteraciones, verbose=True):
        mejor_ruta_global = None
        mejor_distancia_global = float('inf')
        tiempo_inicio = time()
        
        for it in range(iteraciones):
            iter_inicio = time()
            rutas, distancias = self._construir_rutas()
            self._actualizar_feromonas(rutas, distancias)
            
            # Estadísticas de la iteración
            mejor_distancia = min(distancias)
            peor_distancia = max(distancias)
            promedio_distancia = np.mean(distancias)
            
            # Actualizar mejor solución global
            idx_mejor = np.argmin(distancias)
            if mejor_distancia < mejor_distancia_global:
                mejor_distancia_global = mejor_distancia
                mejor_ruta_global = rutas[idx_mejor]
            
            # Registrar estadísticas
            self.historial['mejor_distancia'].append(mejor_distancia)
            self.historial['promedio_distancias'].append(promedio_distancia)
            self.historial['peor_distancia'].append(peor_distancia)
            self.historial['tiempo_iteracion'].append(time() - iter_inicio)
            
            if verbose and (it + 1) % 10 == 0:
                print(f"Iteración {it+1:3d}: Mejor = {mejor_distancia:7.2f} km | "
                      f"Promedio = {promedio_distancia:7.2f} km | "
                      f"Peor = {peor_distancia:7.2f} km | "
                      f"Tiempo = {self.historial['tiempo_iteracion'][-1]:.3f}s")
        
        tiempo_total = time() - tiempo_inicio
        if verbose:
            print("\n" + "="*70)
            print(f"Optimización completada en {iteraciones} iteraciones")
            print(f"Tiempo total: {tiempo_total:.2f} segundos")
            print(f"Tiempo promedio por iteración: {np.mean(self.historial['tiempo_iteracion']):.3f} segundos")
            print("="*70)
        
        return mejor_ruta_global, mejor_distancia_global
    
    def _construir_rutas(self):
        rutas = []
        distancias = []
        
        for _ in range(self.n_hormigas):
            ruta, distancia = self._construir_ruta()
            rutas.append(ruta)
            distancias.append(distancia)
        
        return rutas, distancias
    
    def _construir_ruta(self):
        ruta = []
        distancia_total = 0
        
        # Estrategia de inicio: 50% aleatorio, 50% ciudad más conectada
        if random.random() < 0.5:
            ciudad_actual = random.randint(0, self.n-1)
        else:
            ciudad_actual = np.argmax(np.sum(1/(self.distancias + np.eye(self.n)*1e-10), axis=1))
        
        ruta.append(ciudad_actual)
        ciudades_visitadas = set([ciudad_actual])
        
        while len(ciudades_visitadas) < self.n:
            # Calcular probabilidades con normalización numéricamente estable
            probabilidades = np.zeros(self.n)
            
            for ciudad in range(self.n):
                if ciudad not in ciudades_visitadas:
                    tau = self.feromonas[ciudad_actual][ciudad] ** self.alpha
                    eta = self.visibilidad[ciudad_actual][ciudad] ** self.beta
                    probabilidades[ciudad] = tau * eta
            
            # Manejo de casos donde todas las probabilidades son cero
            if np.sum(probabilidades) == 0:
                # Seleccionar la ciudad no visitada más cercana
                distancias_no_visitadas = np.copy(self.distancias[ciudad_actual])
                distancias_no_visitadas[list(ciudades_visitadas)] = np.inf
                ciudad_siguiente = np.argmin(distancias_no_visitadas)
            else:
                probabilidades /= np.sum(probabilidades)
                ciudad_siguiente = np.random.choice(range(self.n), p=probabilidades)
            
            # Actualizar ruta
            distancia_total += self.distancias[ciudad_actual][ciudad_siguiente]
            ruta.append(ciudad_siguiente)
            ciudades_visitadas.add(ciudad_siguiente)
            ciudad_actual = ciudad_siguiente
        
        # Completar el ciclo
        distancia_total += self.distancias[ruta[-1]][ruta[0]]
        ruta.append(ruta[0])
        
        return ruta, distancia_total
    
    def _actualizar_feromonas(self, rutas, distancias):
        # Evaporación
        self.feromonas *= (1 - self.rho)
        
        # Deposición elitista: solo las mejores hormigas contribuyen
        mejores_hormigas = np.argsort(distancias)[:max(1, int(self.n_hormigas * 0.2))]  # Top 20%
        
        for idx in mejores_hormigas:
            contribucion = self.Q / distancias[idx]
            ruta = rutas[idx]
            
            for i in range(len(ruta)-1):
                ciudad_actual = ruta[i]
                ciudad_siguiente = ruta[i+1]
                self.feromonas[ciudad_actual][ciudad_siguiente] += contribucion
                self.feromonas[ciudad_siguiente][ciudad_actual] += contribucion  # Matriz simétrica
    
    def generar_reporte(self, mejor_ruta, mejor_distancia):
        print("\n" + "="*70)
        print("REPORTE DE OPTIMIZACIÓN - REGIÓN PUNO")
        print("="*70)
        
        # Detalle de la mejor ruta
        print("\nMEJOR RUTA ENCONTRADA:")
        ruta_nombres = [self.ciudades[i] for i in mejor_ruta]
        print(" → ".join(ruta_nombres))
        print(f"\nDistancia total: {mejor_distancia:.2f} km")
        
        # Detalle por tramos
        print("\nDETALLE POR TRAMOS:")
        distancia_acumulada = 0
        for i in range(len(mejor_ruta)-1):
            origen = mejor_ruta[i]
            destino = mejor_ruta[i+1]
            distancia = self.distancias[origen][destino]
            distancia_acumulada += distancia
            print(f"{self.ciudades[origen]:<12} → {self.ciudades[destino]:<12}: "
                  f"{distancia:5.1f} km  (Acumulado: {distancia_acumulada:6.1f} km)")
        
        # Estadísticas de convergencia
        print("\nESTADÍSTICAS DE CONVERGENCIA:")
        print(f"- Mejor distancia inicial: {self.historial['mejor_distancia'][0]:.2f} km")
        print(f"- Mejor distancia final:   {mejor_distancia:.2f} km")
        mejora = (self.historial['mejor_distancia'][0] - mejor_distancia) / self.historial['mejor_distancia'][0] * 100
        print(f"- Porcentaje de mejora:    {mejora:.2f}%")
        print(f"- Iteración de convergencia: {np.argmin(self.historial['mejor_distancia']) + 1}")
        
        # Gráficos
        self._generar_graficos()
    
    def _generar_graficos(self):
        plt.figure(figsize=(15, 5))
        
        # Gráfico de convergencia
        plt.subplot(1, 2, 1)
        plt.plot(self.historial['mejor_distancia'], 'b-', label='Mejor')
        plt.plot(self.historial['promedio_distancias'], 'g--', label='Promedio')
        plt.plot(self.historial['peor_distancia'], 'r:', label='Peor')
        plt.title("Convergencia del Algoritmo ACO\nRegión de Puno, Perú")
        plt.xlabel("Iteración")
        plt.ylabel("Distancia (km)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Gráfico de tiempo por iteración
        plt.subplot(1, 2, 2)
        plt.plot(self.historial['tiempo_iteracion'], 'm-')
        plt.title("Tiempo de Ejecución por Iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Tiempo (s)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Datos de la región de Puno según el informe
def cargar_datos_puno():
    ciudades = ["Puno", "Juliaca", "Ilave", "Juli", "Lampa", "Ayaviri", "Desaguadero"]
    
    distancias = np.array([
        # Puno, Juliaca, Ilave, Juli, Lampa, Ayaviri, Desaguadero
        [   0,   44,   54,   80,   82,  126,  148],  # Puno
        [  44,    0,   98,  124,   32,   84,  192],  # Juliaca
        [  54,   98,    0,   26,  136,  180,   94],  # Ilave
        [  80,  124,   26,    0,  162,  206,  120],  # Juli
        [  82,   32,  136,  162,    0,   52,  224],  # Lampa
        [ 126,   84,  180,  206,   52,    0,  276],  # Ayaviri
        [ 148,  192,   94,  120,  224,  276,    0]   # Desaguadero
    ])
    
    return ciudades, distancias

if __name__ == "__main__":
    # Cargar datos de Puno
    ciudades, distancias = cargar_datos_puno()
    
    print("="*70)
    print("ALGORITMO DE COLONIA DE HORMIGAS OPTIMIZADO")
    print("PROBLEMA DEL VENDEDOR VIAJERO - REGIÓN PUNO, PERÚ")
    print("="*70)
    
    print("\nCiudades incluidas en el estudio:")
    for i, ciudad in enumerate(ciudades):
        print(f"{i+1}. {ciudad}")
    
    # Crear y ejecutar el algoritmo optimizado
    aco = ACO_TSP_Optimizado(distancias, ciudades)
    
    print("\nIniciando optimización con parámetros:")
    print(f"- Número de hormigas: {aco.n_hormigas}")
    print(f"- Alpha (feromonas): {aco.alpha}")
    print(f"- Beta (visibilidad): {aco.beta}")
    print(f"- Rho (evaporación): {aco.rho}")
    print(f"- Q (constante): {aco.Q}")
    print(f"- Iteraciones: 100")
    
    mejor_ruta, mejor_distancia = aco.run(iteraciones=100)
    
    # Generar reporte completo
    aco.generar_reporte(mejor_ruta, mejor_distancia)
    
    # Mostrar matriz de feromonas final
    print("\nMatriz de feromonas final (normalizada):")
    feromonas_norm = aco.feromonas / np.max(aco.feromonas)
    df_feromonas = pd.DataFrame(feromonas_norm, index=ciudades, columns=ciudades)
    print(df_feromonas.style.format("{:.2f}").background_gradient(cmap='YlOrRd'))
