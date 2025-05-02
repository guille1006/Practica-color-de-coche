from collections import defaultdict
import pandas as pd

class Pyurification:
    def __init__(self, df, v_depend=False, col_depend=False, col_cat=[], umbral_cat=0.05):
        self.df = df
        self.len = len(df)
        self.uniques = {}
        self.cols = df.columns
        self.umbral_cat = umbral_cat

        self.posible_null_values = defaultdict(set)


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
                self.summarize_not_num(self.df[col], col)

            else:
                self.col_cat.append(col)

                
               
#---------------------------------------------------------------------------------
    def summarize_not_num(self, series, col):
        tipo = series.dtype
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

        if tipo not in tipos_numericos:
            for elemenent in series:
                if elemenent not in tipos_numericos:
                    self.posible_null_values[col].add(elemenent)

#---------------------------------------------------------------------------------
    def dicotom_to_number(self, series):
        val_unicos = series.unique()
        val_unicos.sort()
        return series.replace({val_unicos[0]:0, val_unicos[1]:1})


#---------------------------------------------------------------------------------
    def show_col_types(self):
        pass


#---------------------------------------------------------------------------------

    def show_frecuencys(self, n_groups_max=250):
        resultados = {}
        frecuencias = []
       
        columns_showed = [x for x in self.uniques if x<n_groups_max]

        for cols in columns_showed:
            pass