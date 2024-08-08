# Datos de ejemplo
clientes = [
    {"id": 1, "nombre": "Juan"},
    {"id": 2, "nombre": "Maria"},
    {"id": 3, "nombre": "Carlos"},
]

productos = [
    {"id": 101, "nombre": "Laptop", "categoria": "Computadoras"},
    {"id": 102, "nombre": "Mouse", "categoria": "Accesorios"},
    {"id": 103, "nombre": "Teclado", "categoria": "Accesorios"},
    {"id": 104, "nombre": "Monitor", "categoria": "Computadoras"},
    {"id": 105, "nombre": "Impresora", "categoria": "Periféricos"},
]

historial_compras = {
    1: [101, 102],  # Juan compró una Laptop y un Mouse
    2: [104],       # Maria compró un Monitor
    3: [101, 103],  # Carlos compró una Laptop y un Teclado
}


import numpy as np
from sklearn.neighbors import NearestNeighbors

# Crear una lista de clientes y productos
clientes_ids = [cliente['id'] for cliente in clientes]
productos_ids = [producto['id'] for producto in productos]

# Crear una matriz de interacciones (clientes x productos)
interacciones = np.zeros((len(clientes), len(productos)))

# Llenar la matriz con las compras registradas
for cliente_id, compras in historial_compras.items():
    for producto_id in compras:
        cliente_idx = clientes_ids.index(cliente_id)
        producto_idx = productos_ids.index(producto_id)
        interacciones[cliente_idx, producto_idx] = 1

print(interacciones)


# Crear el modelo de KNN
modelo_knn = NearestNeighbors(metric='cosine', algorithm='brute')
modelo_knn.fit(interacciones)

def recomendar_accesorios_knn(cliente_id, modelo_knn, interacciones, productos):
    cliente_idx = clientes_ids.index(cliente_id)
    
    # Encontrar los vecinos más cercanos
    distancias, indices = modelo_knn.kneighbors([interacciones[cliente_idx]], n_neighbors=3)
    
    # Identificar productos que los vecinos han comprado y el cliente no
    recomendaciones = set()
    for vecino_idx in indices.flatten():
        if vecino_idx != cliente_idx:
            vecino_compras = np.where(interacciones[vecino_idx] > 0)[0]
            recomendaciones.update(vecino_compras)
    
    # Filtrar para que no recomiende productos que el cliente ya compró
    compras_cliente = np.where(interacciones[cliente_idx] > 0)[0]
    recomendaciones_finales = [productos[idx]['nombre'] for idx in recomendaciones if idx not in compras_cliente]

    return recomendaciones_finales

# Ejemplo de uso
cliente_id = 1
recomendaciones_knn = recomendar_accesorios_knn(cliente_id, modelo_knn, interacciones, productos)
print(f"Recomendaciones (KNN) para el cliente {cliente_id}: {recomendaciones_knn}")


def mostrar_menu():
    print("\n--- Tienda de Electrónica ---")
    print("1. Ver catálogo de productos")
    print("2. Ver historial de compras")
    print("3. Obtener recomendaciones")
    print("4. Salir")

def ver_catalogo(productos):
    print("\n--- Catálogo de Productos ---")
    for producto in productos:
        print(f"{producto['id']}: {producto['nombre']} ({producto['categoria']})")

def ver_historial(historial_compras, clientes, productos):
    print("\n--- Historial de Compras ---")
    for cliente in clientes:
        compras = historial_compras.get(cliente["id"], [])
        productos_comprados = [productos[p_id-101]["nombre"] for p_id in compras]
        print(f"{cliente['nombre']}: {', '.join(productos_comprados)}")

def main():
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            ver_catalogo(productos)
        elif opcion == "2":
            ver_historial(historial_compras, clientes, productos)
        elif opcion == "3":
            cliente_id = int(input("Ingresa el ID del cliente: "))
            recomendaciones_knn = recomendar_accesorios_knn(cliente_id, modelo_knn, interacciones, productos)
            if recomendaciones_knn:
                print(f"Recomendaciones (KNN) para el cliente {cliente_id}: {', '.join(recomendaciones_knn)}")
            else:
                print("No hay recomendaciones disponibles.")
        elif opcion == "4":
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
