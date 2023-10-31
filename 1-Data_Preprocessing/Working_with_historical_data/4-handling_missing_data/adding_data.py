import pandas as pd

# Leer el archivo CSV original
data = pd.read_csv('1-Data_Preprocessing/Working_with_historical_data/4-handling_missing_data/preprocessed_data-historical_data.csv')

# Crear un rango de timestamps entre el mínimo y máximo en el DataFrame original
all_timestamps = pd.DataFrame({'time': range(data['time'].min(), data['time'].max() + 1)})

# Encontrar los timestamps que ya existen en los datos originales
existing_timestamps = set(data['time'])

# Filtrar los nuevos timestamps para evitar duplicados
new_timestamps = all_timestamps[~all_timestamps['time'].isin(existing_timestamps)]

# Crear un DataFrame con los nuevos timestamps y waterMeasured establecido en 0
new_data = pd.DataFrame({
    'time': new_timestamps['time'],
    'station': 'Innova-1',
    'waterMeasured': 0.0
})

# Verificar si los nuevos timestamps ya existen en los datos originales antes de agregarlos
new_data = new_data[~new_data['time'].isin(existing_timestamps)]

# Concatenar los datos originales con los nuevos datos
merged_data = pd.concat([data, new_data], ignore_index=True)

merged_data = merged_data.sort_values(by='waterMeasured', ascending=False)
merged_data = merged_data.drop_duplicates(subset='time', keep='first')

# Encuentra las filas con valores de 'time' duplicados y cuenta cuántos hay
num_valores_time_duplicados = merged_data.duplicated().sum()

# Imprime el número de valores 'time' duplicados
print(f"Número de valores 'time' duplicados: {num_valores_time_duplicados}")

# Ordenar el DataFrame por timestamp
merged_data.sort_values(by='time', inplace=True)

# Guardar el DataFrame actualizado en un nuevo archivo CSV
merged_data.to_csv('tudata_actualizado.csv', index=False)

print("Archivo actualizado correctamente.")


366038