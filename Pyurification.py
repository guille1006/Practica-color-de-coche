from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import re

class Pyurification:
    def __init__(self, df, v_depend=False, col_depend=False, col_cat=[], umbral_cat=0.05):
        self.df = df
        self.len = len(df)
        self.uniques = {}
        self.cols = df.columns
        self.umbral_cat = umbral_cat

        self.posible_null_values = defaultdict(set)
        self.extra_elements = defaultdict(set)


# Guardado de las variables dependientes e independientes -------------------------------------
        if v_depend:
            self.v_depend = v_depend
            self.col_depend = None
            self.v_independ = df
            self.col_depend = df.columns

        elif col_depend:
            self.col_depend = col_depend
            self.v_depend = df[col_depend]
            self.col_independ = [x for x in df.columns if x not in col_depend]    #Posible error
            self.v_independ = df[self.col_independ]

        else:
            self.col_depend = None
            self.v_depend = None
            self.col_independ = df.columns
            self.v_independ = df

# Guardado de cada tipo de variable mediante el metodo select_type_variables()
        self.col_dicotom = []
        self.col_cat = []
        self.col_num = []

        self.select_type_variables()
#---------------------------------------------------------------------------------
    def select_type_variables(self):
        # Ademas de guardar cada columna en su tipo de variable, guarda clos valores unicos en self.uniques
        col_to_analyze = self.cols
       
        for col in col_to_analyze:
            serie = self.v_independ[col]
            self.uniques[col] = serie.nunique()

            if self.uniques[col] == 2:
                self.col_dicotom.append(col)
                self.df[col] = self.dicotom_to_number(self.df[col])

            elif self.uniques[col] > self.umbral_cat*self.len:
                self.col_num.append(col)
                self.summarize_not_num(col)

            else:
                self.col_cat.append(col)

                
               
#---------------------------------------------------------------------------------
    def summarize_not_num(self, col):
        tipo = self.df[col].dtype
        tipos_numericos = [
            # Tipos nativos de Python
            int,
            float,
            complex,

            # Tipos NumPy (enteros con y sin signo)
            'int8', 'int16', 'int32', 'int64',
            'uint8', 'uint16', 'uint32', 'uint64',

            # Tipos NumPy (decimales)
            'float16', 'float32', 'float64', 'float128',  # float128 no siempre disponible

            # Tipos NumPy (complejos)
            'complex64', 'complex128', 'complex256',      # complex256 depende del sistema
        ]
        if self.posible_null_values[col]:
            del self.posible_null_values[col]

        if tipo not in tipos_numericos:
            if self.posible_null_values[col]:
                del self.posible_null_values[col]
            for elemenent in self.df[col]:
                if elemenent not in tipos_numericos:
                    try:
                        float(elemenent)
                    except ValueError:
                        self.posible_null_values[col].add(elemenent)
#---------------------------------------------------------------------------------
    def change_to_num(self, columnas):
        if not columnas:
            print("Debe añadir alguna columna valida")
            return
        
        for col in columnas:
            if col not in self.col_cat:
                print(f"La columna {col} no se encuentra en las variables categoricas de los datos inicales")
                continue
            else:
                self.col_cat.remove(col)
                self.col_num.append(col)
                self.summarize_not_num(col)

#---------------------------------------------------------------------------------
    def dicotom_to_number(self, series):
        val_unicos = series.unique()
        val_unicos.sort()
        return series.replace({val_unicos[0]:0, val_unicos[1]:1})

#---------------------------------------------------------------------------------
    def remove_string_from_num(self, col):
        # Esto solo funciona con numeros seguidos de letras.
        self.find_string_in_num(col)

        for extra in self.extra_elements[col]:
            patron = r'\s*' + re.escape(extra) + r'$'
            # Primero limpia el texto
            self.df[col] = self.df[col].astype(str).str.replace(patron, "", regex=True)

            # Luego convierte a float
            self.df[col] = pd.to_numeric(self.df[col])  # Convertirá valores no convertibles a NaN


        self.uniques[col] = self.df[col].nunique()
        self.summarize_not_num(col)


#---------------------------------------------------------------------------------
    def create_variable_dicotomica_from_extra_string(self, col):
        self.find_string_in_num(col)

        for extra in self.extra_elements[col]:
            self.df[extra] = self.df[col].astype(str).str.contains(extra).astype(int)
            self.col_dicotom.append(extra)
            self.uniques[extra] = self.df[extra].nunique()
        
        self.cols = self.df.columns

#---------------------------------------------------------------------------------
    def create_and_remove_string_from_num(self, col):
        self.create_variable_dicotomica_from_extra_string(col)
        self.remove_string_from_num(col)
#---------------------------------------------------------------------------------
    def find_string_in_num(self,col):
        # Esto solo funciona con numeros seguidos de letras.
        if self.extra_elements[col]:
            del self.extra_elements[col]

        for elements in self.df[col]:
            match = re.match(r'^\d+(\.\d+)?\s*([a-zA-Z]+)$', elements)
            if match:
                numeros = match.group(1)  # '123'
                letras = match.group(2)   # 'abc'
                self.extra_elements[col].add(letras)

#---------------------------------------------------------------------------------
    def change_value_in_col(self, col, inicial_values=None, condicional_values=None, changed_values=np.NaN):
        if not inicial_values and not condicional_values:
            print("No se ha realizado ningun cambio, indica que valores quieres cambiar")
        
        if inicial_values:
            for value in inicial_values:
                self.df[col] = self.df[col].replace(value, changed_values)
        
        if condicional_values:
            for value in condicional_values:
                value = value.items()

                if value[0]=="$gt":
                    self.df[col] = self.df[col].where(self.df[col]>value, changed_values)
                elif value[0]=="$gte":
                    self.df[col] = self.df[col].where(self.df[col]>=value, changed_values)
                elif value[0]=="$lt":
                    self.df[col] = self.df[col].where(self.df[col]<value, changed_values)
                elif value[0]=="$lte":
                    self.df[col] = self.df[col].where(self.df[col]<=value, changed_values)

        self.uniques[col]=self.df[col].nunique()
        self.summarize_not_num(col)


#---------------------------------------------------------------------------------
    def onehotencoder_col(self, col):
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded = encoder.fit_transform(self.df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))

        # Elimino la columna original
        self.df.drop(columns=[col], inplace=True)

        # Guardo las columnas onehotencoded en el df
        self.df = pd.concat([self.df, encoded_df], axis=1)
#---------------------------------------------------------------------------------
    def onehotencoder(self, cols=None):
        if not cols:
            cols=self.col_cat

        for col in cols:
            self.onehotencoder_col(col)
#---------------------------------------------------------------------------------
    def show_col_types(self):
        col_types = defaultdict(list)
        for col in self.cols:
            # Fila "Tipo"
            if col in self.col_dicotom:
                col_types[col].append("Dicotomica")
            elif col in self.col_cat:
                col_types[col].append("Categoríca")
            elif col in self.col_num:
                col_types[col].append("Numeríca")
            # Fila "Tipo asignado por pandas"
            col_types[col].append(self.df[col].dtype)
            # Fila "Valores unicos"
            col_types[col].append(self.uniques[col])
            # Fila "Posibles errores"
            if self.posible_null_values[col]:
                if len(self.posible_null_values[col])>10:
                   col_types[col].append(f"{len(self.posible_null_values[col])} valores distintos a un numerico. Recomiendo revisión de tipo de variable")
                else: 
                    col_types[col].append(self.posible_null_values[col])
            else: 
                col_types[col].append(None)

        show = pd.DataFrame(col_types)
        show.index = ["Tipo", "Tipo asignado por pandas", "Valores unicos", "Posibles errores"]
        return show


#---------------------------------------------------------------------------------

    def show_frecuencys(self):
        resultados = []
       
        columns_showed = self.col_cat
        for col in columns_showed:
            serie_con_cada_frecuencia = self.df[col].value_counts()
            serie_con_porcentaje = self.df[col].value_counts(normalize=True)

            resumen = pd.DataFrame({"n": serie_con_cada_frecuencia,
                                    "%": serie_con_porcentaje})
            
            resultados.append(resumen)

        final = pd.concat(resultados, keys=columns_showed)
        return final