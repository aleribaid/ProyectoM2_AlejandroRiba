"""
FleetLogix - Generación de Datos Sintéticos
Genera 505000+ registros respetando relaciones y reglas de negocio
"""

import psycopg2
from psycopg2.extras import execute_batch
# NOTA: Se eliminó 'pandas' y 'sqlalchemy' que no eran necesarios o generaban inconsistencia
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import logging
from tqdm import tqdm
import json
import sys


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_generation.log'),
        logging.StreamHandler()
    ]
)

# Configuración de conexión
DB_CONFIG = {
    'host': 'localhost',
    'database': 'FLEETLOGIX DATABASE',
    'user': 'postgres',
    'password': 'alikavelero',  # Cambiar por tu contraseña
    'port': 5432
}

# Inicializar Faker con semilla para reproducibilidad
fake = Faker('es_AR')  # Español Argentina
Faker.seed(42)
random.seed(42)
np.random.seed(42)

class DataGenerator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
        self.cursor = None
        self.cities = ['Buenos Aires', 'Rosario', 'Cordoba', 'San Miguel de Tucuman', 'Mendoza']
        
        # Contadores para logging (ahora es la fuente única de verdad)
        self.counters = {
            'vehicles': 200,
            'drivers': 400,
            'routes': 50,
            'trips': 100000,
            'deliveries': 400000,
            'maintenance': 5000
        }
    
    def connect(self):
        """Establecer conexión con la base de datos"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            logging.info(" Conexión exitosa a PostgreSQL")
            return True
        except Exception as e:
            logging.error(f" Error al conectar: {e}")
            return False
   
    
    def truncate_tables(self):
        """Truncar tablas en orden para evitar conflictos de FK"""
        logging.warning("Iniciando truncado de tablas (CASCADE)...")
        tables_to_truncate = [
            'maintenance', 'deliveries', 'trips', 
            'routes', 'drivers', 'vehicles'
        ]
        # El orden es de "hijos" a "padres"
        try:
            for table in tables_to_truncate:
                # Usar el cursor de la clase
                self.cursor.execute(f'TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;')
            self.connection.commit()
            logging.info("Tablas truncadas y contadores reiniciados.")
        
        except Exception as e:
            logging.error(f"Error en truncado: {e}")
            self.connection.rollback() # Usar el rollback de la clase
            sys.exit(1) # Salir si no podemos truncar

    # ---------- GENERACIÓN DE DATOS ----------
    
    # MEJORA 2: Usar 'self.counters' como valor por defecto
    def generate_vehicles(self):
        """Generar vehículos con diferentes tipos y capacidades"""
        count = self.counters['vehicles'] # Usar el contador de la clase
        logging.info(f"Generando {count} vehículos...")
        
        vehicle_types = [
            ('Camión Grande', 5000, 'diesel', 0.3),
            ('Camión Mediano', 3000, 'diesel', 0.3),
            ('Van', 1500, 'gasolina', 0.3),
            ('Motocicleta', 50, 'gasolina', 0.1)
        ]
        
        vehicles = []
        for i in range(count):
            v_type, capacity, fuel, prob = random.choices(
                vehicle_types, 
                weights=[vt[3] for vt in vehicle_types]
            )[0]
            
            license_plate = f"{fake.random_uppercase_letter()}{fake.random_uppercase_letter()}{fake.random_uppercase_letter()}{random.randint(100,999)}"
            acquisition_date = fake.date_between(start_date='-5y', end_date='-1m')
            status = random.choice(['active'] * 9 + ['maintenance'])
            
            vehicles.append((
                license_plate,
                v_type,
                capacity,
                fuel,
                acquisition_date,
                status
            ))
        
        # Insertar en batch
        query = """INSERT INTO vehicles (license_plate, vehicle_type, capacity_kg, fuel_type, acquisition_date, status)
            VALUES (%s, %s, %s, %s, %s, %s)"""
         
        execute_batch(self.cursor, query, vehicles, page_size=100)
        self.connection.commit()
        # self.counters['vehicles'] = count # Ya no es necesario, se lee de ahí
        logging.info(f" {count} vehículos insertados")
    
    def generate_drivers(self):
        """Generar conductores con datos realistas"""
        count = self.counters['drivers'] # Usar el contador de la clase
        logging.info(f"Generando {count} conductores...")
        
        drivers = []
        license_types = ['C1', 'C2', 'C3', 'A2']  # Tipos de licencia Argentina
        
        for i in range(count):
            employee_code = f"EMP{str(i+1).zfill(4)}"
            first_name = fake.first_name()
            last_name = fake.last_name()
            license_number = f"{random.randint(1000000000, 9999999999)}"
            license_type = random.choice(license_types)
            license_expiry = fake.date_between(start_date='-1m', end_date='+3y')
            phone = f"3{random.randint(100000000, 999999999)}"
            hire_date = fake.date_between(start_date='-5y', end_date='-1w')
            status = random.choice(['active'] * 19 + ['inactive'])
            
            drivers.append((
                employee_code,
                first_name,
                last_name,
                license_number,
                license_type, 
                license_expiry,
                phone,
                hire_date,
                status
            ))
        
        query = """
            INSERT INTO drivers (employee_code, first_name, last_name, 
                               license_number, license_type, license_expiry, phone, 
                               hire_date, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        execute_batch(self.cursor, query, drivers, page_size=100)
        self.connection.commit()
        logging.info(f" {count} conductores insertados")
    
    def generate_routes(self):
        """Generar rutas entre las 5 ciudades principales"""
        count = self.counters['routes'] # Usar el contador de la clase
        logging.info(f"Generando {count} rutas...")
        
        routes = []
        route_counter = 1
        
        for origin in self.cities:
            for destination in self.cities:
                if origin != destination:
                    num_routes = 3 if origin == 'Buenos Aires' or destination == 'Buenos Aires' else 2
                    
                    for i in range(num_routes):
                        route_code = f"R{str(route_counter).zfill(3)}"
                        base_distance = self._get_distance(origin, destination)
                        distance = base_distance + random.uniform(-100, 100)
                        avg_speed = random.uniform(60, 80)
                        duration = distance / avg_speed
                        toll_cost = int(distance / 100) * 15000
                        
                        routes.append((
                            route_code,
                            origin,
                            destination,
                            round(distance, 2),
                            round(duration, 2),
                            toll_cost
                        ))
                        
                        route_counter += 1
                        if route_counter > count:
                            break
                    
                    if route_counter > count:
                        break
            
            if route_counter > count:
                break
        
        routes = routes[:count]
        
        query = """ INSERT INTO routes (route_code, origin_city, destination_city, distance_km, estimated_duration_hours, toll_cost) VALUES (%s, %s, %s, %s, %s, %s)"""
        
        execute_batch(self.cursor, query, routes, page_size=50)
        self.connection.commit()
        logging.info(f" {count} rutas insertadas")
    
    def _get_distance(self, origin, destination):
        """Obtener distancia aproximada entre ciudades Argentinas""" 
        distances = {
            ('Buenos Aires', 'Rosario'): 299,
            ('Buenos Aires', 'Cordoba'): 696,
            ('Buenos Aires', 'San Miguel de Tucuman'): 1247,
            ('Buenos Aires', 'Mendoza'): 1050,
            ('Rosario', 'Cordoba'): 400,
            ('Rosario', 'San Miguel de Tucuman'): 952,
            ('Rosario', 'Mendoza'): 873,
            ('Cordoba', 'San Miguel de Tucuman'): 560,
            ('Cordoba', 'Mendoza'): 600,
            ('San Miguel de Tucuman', 'Mendoza'): 958
        }
        
        key = tuple(sorted([origin, destination]))
        return distances.get(key, 500)
    
    def generate_trips(self):
        """Generar viajes en 2 años de operación (Eficiente en memoria)"""
        count = self.counters['trips'] # Usar el contador de la clase
        logging.info(f"Generando {count} viajes...")
        
        # Obtener IDs válidos
        self.cursor.execute("SELECT vehicle_id, capacity_kg FROM vehicles WHERE status = 'active'")
        vehicles = self.cursor.fetchall()
        self.cursor.execute("SELECT driver_id FROM drivers WHERE status = 'active'")
        drivers = [d[0] for d in self.cursor.fetchall()]
        self.cursor.execute("SELECT route_id, distance_km, estimated_duration_hours FROM routes")
        routes = self.cursor.fetchall()
        
        start_date = datetime.now() - timedelta(days=730)
        
        # MEJORA : Patrón de generación e inserción por lotes
        trips_batch = [] # Lote temporal
        batch_size = 1000 # Tamaño del lote a insertar
        current_date = start_date
        
        query = """
            INSERT INTO trips (vehicle_id, driver_id, route_id, departure_datetime,
                               arrival_datetime, fuel_consumed_liters, total_weight_kg, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for i in tqdm(range(count), desc="Generando trips"):
            vehicle_id, capacity = random.choice(vehicles)
            capacity = float(capacity)
            driver_id = random.choice(drivers)
            route_id, distance, est_duration = random.choice(routes)
            
            distance = float(distance)
            est_duration = float(est_duration)
            
            hour = np.random.choice(
                range(24), 
                p=self._get_hourly_distribution()
            )
            departure = current_date.replace(hour=hour, minute=random.randint(0, 59))
            
            actual_duration = est_duration * random.uniform(0.8, 1.3)
            arrival = departure + timedelta(hours=actual_duration)
            
            fuel_consumed = distance * random.uniform(0.08, 0.15)
            total_weight = capacity * random.uniform(0.4, 0.9)
            
            if arrival < datetime.now():
                status = 'completed'
            else:
                status = 'in_progress'
            
            trips_batch.append((
                vehicle_id,
                driver_id,
                route_id,
                departure,
                arrival if status == 'completed' else None,
                round(fuel_consumed, 2),
                round(total_weight, 2),
                status
            ))
            
            current_date += timedelta(minutes=int(1440 * 2 * 365 / count))
            
            # Insertar el lote cuando esté lleno
            if len(trips_batch) >= batch_size:
                execute_batch(self.cursor, query, trips_batch, page_size=batch_size)
                self.connection.commit()
                trips_batch.clear()
                
                if (i + 1) % 10000 == 0:
                    logging.info(f"  Progreso: {i+1}/{count} trips insertados")
        
        # Insertar el lote restante
        if trips_batch:
            execute_batch(self.cursor, query, trips_batch, page_size=len(trips_batch))
            self.connection.commit()
            trips_batch.clear()
        
        logging.info(f" {count} viajes insertados")
    
    def _get_hourly_distribution(self):
        """Distribución de probabilidad por hora del día"""
        probs = np.ones(24) * 0.02
        probs[6:20] = 0.06
        probs[8:12] = 0.08
        probs[14:18] = 0.07
        return probs / probs.sum()
    
    def generate_deliveries(self):
        """Generar entregas (promedio 4 por viaje) (Eficiente en memoria)"""
        count = self.counters['deliveries'] # Usar el contador de la clase
        logging.info(f"Generando {count} entregas...")
        
        # Obtener viajes válidos
        self.cursor.execute("""
            SELECT 
                t.trip_id, 
                t.departure_datetime, 
                t.arrival_datetime, 
                t.total_weight_kg, 
                r.destination_city
            FROM trips t
            JOIN routes r ON t.route_id = r.route_id
            WHERE t.status = 'completed' OR t.status = 'in_progress'
        """)
        
        trips_data = self.cursor.fetchall()
        
        
        deliveries_batch = []
        batch_size = 1000 # Tamaño del lote de entregas
        delivery_counter = 0
        
        query = """
            INSERT INTO deliveries (trip_id, tracking_number, customer_name,
                                  delivery_address, package_weight_kg, scheduled_datetime,
                                  delivered_datetime, delivery_status, recipient_signature)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for trip_id, departure, arrival, total_weight, city in tqdm(trips_data, desc="Generando deliveries"):
            num_deliveries = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            
            # Evitar error si el peso es None o 0
            if not total_weight or total_weight <= 0:
                continue
                
            weights = self._distribute_weight(float(total_weight), num_deliveries)
            
            if arrival:
                delivery_duration = (arrival - departure).total_seconds() / 3600
                time_per_delivery = delivery_duration / num_deliveries
            else:
                time_per_delivery = 0.5
            
            for i in range(num_deliveries):
                tracking_number = f"FL{datetime.now().year}{str(delivery_counter+1).zfill(8)}"
                customer_name = fake.name()
                delivery_address = f"{fake.street_address()}, {city}"
                package_weight = float(weights[i])
                
                scheduled = departure + timedelta(hours=time_per_delivery * (i + 0.5))
                
                if arrival:
                    if random.random() < 0.9:
                        delivered = scheduled + timedelta(minutes=random.randint(-30, 30))
                    else:
                        delivered = scheduled + timedelta(minutes=random.randint(60, 180))
                    
                    delivery_status = 'delivered'
                    signature = random.random() < 0.95
                else:
                    delivered = None
                    delivery_status = 'pending'
                    signature = False
                
                deliveries_batch.append((
                    trip_id,
                    tracking_number,
                    customer_name,
                    delivery_address,
                    round(package_weight, 2),
                    scheduled,
                    delivered,
                    delivery_status,
                    signature
                ))
                
                delivery_counter += 1
                
                # Insertar lote cuando esté lleno
                if len(deliveries_batch) >= batch_size:
                    execute_batch(self.cursor, query, deliveries_batch, page_size=batch_size)
                    self.connection.commit()
                    deliveries_batch.clear()
                    
                    if delivery_counter % 50000 < batch_size:
                         logging.info(f"  Progreso: {delivery_counter}/{count} deliveries insertados")

                if delivery_counter >= count:
                    break
            
            if delivery_counter >= count:
                break
        
        # Insertar el lote restante
        if deliveries_batch:
            execute_batch(self.cursor, query, deliveries_batch[:(count - delivery_counter + len(deliveries_batch))], page_size=len(deliveries_batch))
            self.connection.commit()
            deliveries_batch.clear()
        
        self.counters['deliveries'] = delivery_counter # Actualizar con el conteo real
        logging.info(f" {delivery_counter} entregas insertadas")
    
    def _distribute_weight(self, total_weight, num_packages):
        """Distribuir peso total entre paquetes de manera realista"""
        weights = np.random.exponential(scale=1.0, size=num_packages)
        weights = weights / weights.sum() * total_weight * 0.95
        weights = np.maximum(weights, 0.5)
        return weights
    
    def generate_maintenance(self):
        """Generar registros de mantenimiento"""
        count = self.counters['maintenance']
        logging.info(f"Generando {count} registros de mantenimiento...")

        self.cursor.execute("""
            SELECT 
                v.vehicle_id, 
                v.vehicle_type, 
                COUNT(t.trip_id) as trip_count, 
                MIN(t.departure_datetime) as first_trip, 
                MAX(t.arrival_datetime) as last_trip
            FROM vehicles v
            LEFT JOIN trips t ON v.vehicle_id = t.vehicle_id
            GROUP BY v.vehicle_id, v.vehicle_type
        """)
        vehicle_stats = self.cursor.fetchall()

        maintenance_types = [
            ('Cambio de aceite', 150000, 30),
            ('Revisión de frenos', 250000, 60),
            ('Cambio de llantas', 450000, 90),
            ('Mantenimiento general', 350000, 45),
            ('Revisión de motor', 500000, 60),
            ('Alineación y balanceo', 180000, 30)
        ]
        
        maintenance_records = []

        for vehicle_id, vehicle_type, trip_count, first_trip, last_trip in vehicle_stats:
            
            # --- INICIO DE LA CORRECCIÓN ---
            first_trip_date = None
            last_trip_date = None

            if trip_count == 0 or not first_trip or not last_trip:
                # Caso B: El vehículo no tiene viajes, usamos la fecha de adquisición
                self.cursor.execute("SELECT acquisition_date FROM vehicles WHERE vehicle_id = %s", (vehicle_id,))
                first_trip_date = self.cursor.fetchone()[0] # Ya es un objeto .date
                last_trip_date = datetime.now().date()      # Ya es un objeto .date
                trip_count = 1
            else:
                # Caso A: El vehículo tiene viajes, extraemos el .date() del datetime
                first_trip_date = first_trip.date()
                last_trip_date = last_trip.date()
            
            # Ahora 'last_trip_date' y 'first_trip_date' son siempre objetos .date
            operation_days = (last_trip_date - first_trip_date).days
            # --- FIN DE LA CORRECCIÓN ---

            if operation_days <= 0:
                operation_days = 1
                
            num_maintenance = max(1, trip_count // 20)

            for i in range(min(num_maintenance, count - len(maintenance_records))):
                days_offset = int(operation_days * (i + 1) / (num_maintenance + 1))
                
                # --- CORRECCIÓN ADICIONAL ---
                # Usamos 'first_trip_date' que ya es un objeto .date
                maintenance_date = (first_trip_date + timedelta(days=days_offset))
                
                maint_type, base_cost, days_next = random.choice(maintenance_types)
                cost = base_cost * random.uniform(0.8, 1.2)
                description = f"{maint_type} programado para {maintenance_date.strftime('%Y-%m-%d')}"
                next_maintenance = maintenance_date + timedelta(days=days_next)
                performed_by = f"{fake.first_name()} {fake.last_name()}"
                
                maintenance_records.append((
                    vehicle_id,
                    maintenance_date,
                    maint_type,
                    description,
                    round(cost, 2),
                    next_maintenance,
                    performed_by
                ))
                
                if len(maintenance_records) >= count:
                    break
            
            if len(maintenance_records) >= count:
                break
        
        query = """
            INSERT INTO maintenance (vehicle_id, maintenance_date, maintenance_type,
                                   description, cost, next_maintenance_date, performed_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        execute_batch(self.cursor, query, maintenance_records[:count], page_size=100)
        self.connection.commit()
        self.counters['maintenance'] = min(len(maintenance_records), count)
        logging.info(f" {self.counters['maintenance']} mantenimientos insertados")
    
    def validate_data_quality(self):
        """Validar integridad y calidad de datos"""
        logging.info("\n VALIDANDO CALIDAD DE DATOS...")
        
        validations = {
            "Integridad referencial - Trips sin vehículo válido": """
                SELECT COUNT(*) FROM trips t 
                LEFT JOIN vehicles v ON t.vehicle_id = v.vehicle_id 
                WHERE v.vehicle_id IS NULL
            """,
            "Integridad referencial - Deliveries sin trip válido": """
                SELECT COUNT(*) FROM deliveries d 
                LEFT JOIN trips t ON d.trip_id = t.trip_id 
                WHERE t.trip_id IS NULL
            """,
            "Consistencia temporal - Trips con arrival < departure": """
                SELECT COUNT(*) FROM trips 
                WHERE arrival_datetime IS NOT NULL 
                AND arrival_datetime < departure_datetime
            """,
            "Consistencia de peso - Trips excediendo capacidad": """
                SELECT COUNT(t.trip_id) FROM trips t 
                JOIN vehicles v ON t.vehicle_id = v.vehicle_id 
                WHERE t.total_weight_kg > v.capacity_kg
            """,
            "Entregas sin tracking number": """
                SELECT COUNT(*) FROM deliveries 
                WHERE tracking_number IS NULL OR tracking_number = ''
            """
        }
        
        all_valid = True
        for description, query in validations.items():
            self.cursor.execute(query)
            count = self.cursor.fetchone()[0]
            if count > 0:
                logging.warning(f" [FALLO] {description}: {count} registros")
                all_valid = False
            else:
                logging.info(f"   [OK] {description}: OK")
        
        return all_valid
    
    def generate_summary_report(self):
        """Generar reporte resumen de datos generados"""
        logging.info("\n RESUMEN DE GENERACIÓN DE DATOS")
        logging.info("="*50)
        
        tables = ['vehicles', 'drivers', 'routes', 'trips', 'deliveries', 'maintenance']
        total_records = 0
        
        # Usar los contadores de la clase para el reporte JSON
        # Pero verificar la BD para el log
        final_counts = {}
        
        for table in tables:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = self.cursor.fetchone()[0]
            logging.info(f"  {table}: {count:,} registros")
            total_records += count
            final_counts[table] = count # Guardar el conteo real de la BD
        
        logging.info(f"\n  TOTAL: {total_records:,} registros")
        
        self.cursor.execute("""
            SELECT 
                AVG(delivery_count) as avg_deliveries_per_trip,
                MIN(delivery_count) as min_deliveries,
                MAX(delivery_count) as max_deliveries
            FROM (
                SELECT trip_id, COUNT(*) as delivery_count
                FROM deliveries
                GROUP BY trip_id
            ) as delivery_stats
        """)
        
        avg_del, min_del, max_del = self.cursor.fetchone()
        logging.info(f"\n  Entregas por viaje: AVG={avg_del:.1f}, MIN={min_del}, MAX={max_del}")
        
        # Validar datos ANTES de guardar el resumen
        validation_passed = self.validate_data_quality()
        
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_records': total_records,
            'table_counts': final_counts, # Usar los conteos reales
            'validations_passed': validation_passed
        }
        
        with open('generation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info("\n Resumen guardado en generation_summary.json")
    
    def close(self):
        """Cerrar conexión"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logging.info("\n Conexión cerrada")


def main():
    """Función principal"""
    print(" FLEETLOGIX - Generación de Datos Masivos")
    print("="*60)
    print(f"Objetivo: Generar ~{sum(DataGenerator(DB_CONFIG).counters.values()):,} registros manteniendo integridad")
    print("="*60)
    
    generator = DataGenerator(DB_CONFIG)
    
    try:
        if not generator.connect():
            return
        
        # MEJORA 6: Llamar a truncate_tables
        generator.truncate_tables()
        
        # MEJORA 7: Llamar a los métodos sin parámetros
        # Generar datos en orden (respetando foreign keys)
        generator.generate_vehicles()
        generator.generate_drivers()
        generator.generate_routes()
        generator.generate_trips()
        generator.generate_deliveries()
        generator.generate_maintenance()
        
        # Validar y generar reporte
        generator.generate_summary_report()
        
    except Exception as e:
        logging.error(f" Error fatal durante la generación: {e}")
        # Intentar hacer rollback
        try:
            generator.connection.rollback()
            logging.info("Rollback de transacción realizado.")
        except Exception as rb_e:
            logging.error(f"Error durante el rollback: {rb_e}")
    
    finally:
        generator.close()


if __name__ == "__main__":
    main()