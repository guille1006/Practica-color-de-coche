from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def main():

    # Limpieza de variables
    df = pd.read_excel("datos_tarea25.xlsx")
    df["Levy"] = df["Levy"].replace("-", np.nan)
    df["Levy -"] = df["Levy"].isna()
    df["Levy"] = pd.to_numeric(df["Levy"], errors="coerce")
    mediana_levy = df["Levy"].median()
    df["Levy"] = df["Levy"].fillna(mediana_levy)

    pyrf = Pyurification(df)

    pyrf.change_to_num(["Prod. year", "Engine volume", "Airbags"])
    pyrf.remove_string_from_num("Mileage")
    pyrf.create_and_remove_string_from_num("Engine volume")
    pyrf.onehotencoder(["Manufacturer", "Category", "Fuel type", "Drive wheels"], drop=None)


    # División de variables
    X = pyrf.df
    y = X["Color"]
    X.drop(columns="Color", inplace=True)

    test_size = 0.2
    random_state = 777

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"La frecuencia de cada clase en train es: \n{y_train.value_counts(normalize=True)}")
    print(f"La frecuencia de cada clase en test es: \n{y_test.value_counts(normalize=True)}")


    # Eelccion de variables con Random Forest
    model_feature = RandomForestClassifier(random_state=42)
    model_feature.fit(X, y)

    importancia = pd.Series(model_feature.feature_importances_, index=X.columns)
    importancia = importancia.sort_values(ascending=False)

    # Ordenamos el DataFrame por importancia en orden descendente
    df_importancia = pd.DataFrame({"Variable":model_feature.feature_names_in_, "Importancia":model_feature.feature_importances_}).sort_values(by="Importancia", ascending=False)

    # Creamos el grafico de barras
    plt.bar(df_importancia["Variable"], df_importancia["Importancia"], color="skyblue")
    plt.xlabel("Variable")
    plt.ylabel("Importancia")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Calcular importancia acumulada
    importancia_acumulada = importancia.cumsum()

    importancia_df = pd.DataFrame({
        'Importancia': importancia,
        'Importancia Acumulada': importancia_acumulada
    })
    print(importancia_df)
    variables_importantes = list(importancia.index[:12])

    # ---------------------------------------------------------------------------
    # Arboles de clasificacion
    tree = DecisionTreeClassifier(criterion="entropy",
                              max_depth=10,
                              random_state=random_state)

    tree.fit(X_train, y_train)

    # Ordenamos el DataFrame por importancia en orden descendente
    df_importancia = pd.DataFrame({"Variable":tree.feature_names_in_, "Importancia":tree.feature_importances_}).sort_values(by="Importancia", ascending=False)

    # Creamos el grafico de barras
    plt.bar(df_importancia["Variable"], df_importancia["Importancia"], color="skyblue")
    plt.xlabel("Variable")
    plt.ylabel("Importancia")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Tuneo
    params={
        "max_depth":[2, 3, 5, 7],
        "min_samples_split":[5, 10, 20, 50, 100],
        "criterion": ["gini", "entropy"]
    }
    scoring_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


    #Cross Validation
    decision_tree = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=decision_tree,
                            param_grid=params,
                            cv=4,
                            scoring=scoring_metrics,
                            refit="accuracy")
    grid_search.fit(X_train, y_train)


    # Obtenemos los resultados del grid search.
    results = pd.DataFrame(grid_search.cv_results_)

    results.sort_values(by="mean_test_accuracy", ascending=False, inplace=True)
    results[["params", "mean_test_accuracy", "mean_test_precision_macro", "mean_test_recall_macro", "mean_test_f1_macro"]].head(5)

    # Usaremos unos cuantos modelo
    res_1 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[0]
    res_2 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[1]
    res_3 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[2]
    res_4 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[3]
    res_5 = results[['split0_test_accuracy', 'split1_test_accuracy','split2_test_accuracy', 'split3_test_accuracy']].iloc[4]

    # Creamos una boxplot para los valores de accuracy
    plt.boxplot([res_1.values, res_2.values, res_3.values, res_4.values, res_5.values], 
                labels=["res 1", "res 2", "res 3", "res 4", "res 5"])
    plt.show()

    # Nos quedamos con el modelo 1
    mejor_modelo = results.iloc[0]["params"]

    # Empezamos a predecir
    mejor_arbol = DecisionTreeClassifier(**mejor_modelo)
    mejor_arbol.fit(X_train, y_train)
    print(mejor_arbol.get_params)


    # Predicciones en train y test
    y_train_predict = mejor_arbol.predict(X_train)
    y_test_predict = mejor_arbol.predict(X_test)

    # Medidas de bondad de ajuste en train
    conf_matrix = confusion_matrix(y_train, y_train_predict)
    print(conf_matrix)
    print("Medidas de desempeño")
    print(classification_report(y_train, y_train_predict))


    # Creamos el arbol
    plt.figure(figsize=(20,15))
    plot_tree(mejor_arbol, feature_names=X.columns.tolist(), filled=True, proportion=True)
    plt.show()

    tree_rules = export_text(mejor_arbol, feature_names=list(X.columns), show_weights=True)
    print(tree_rules)


    # Ordenamos el DataFrame por importancia en orden descendente
    df_importancia = pd.DataFrame({"Variable":mejor_arbol.feature_names_in_, "Importancia":mejor_arbol.feature_importances_}).sort_values(by="Importancia", ascending=False)

    # Creamos el grafico de barras
    plt.bar(df_importancia["Variable"], df_importancia["Importancia"], color="skyblue")
    plt.xlabel("Variable")
    plt.ylabel("Importancia")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()



    # -----------------------------------------------------------------------------------------------
    # Random Forest
    X = X[variables_importantes]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"La frecuencia de cada clase en train es: \n{y_train.value_counts(normalize=True)}")
    print(f"La frecuencia de cada clase en test es: \n{y_test.value_counts(normalize=True)}")

    
    RF_model = RandomForestClassifier()
    # Tuneo de los hiperparametros
    params = {
        'n_estimators' : [50, 70, 80, 100],
        'max_depth': [2, 3, 5, 10, 20],
        'bootstrap': [True, False],
        'min_samples_leaf' : [3,10,30],
        'min_samples_split': [5, 10, 20, 50, 100],
        'criterion': ["gini", "entropy"]
    }

    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    # cv = crossvalidation
    grid_search_RF = GridSearchCV(estimator=RF_model, 
                            param_grid=params, 
                            cv=4, 
                            scoring = scoring_metrics, 
                            refit='accuracy')
    grid_search_RF.fit(X_train, y_train)













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
    def onehotencoder_col(self, col, drop="first"):
        encoder = OneHotEncoder(sparse_output=False, drop=drop)
        encoded = encoder.fit_transform(self.df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))

        # Elimino la columna original
        self.df.drop(columns=[col], inplace=True)

        # Guardo las columnas onehotencoded en el df
        self.df = pd.concat([self.df, encoded_df], axis=1)
#---------------------------------------------------------------------------------
    def onehotencoder(self, cols=None, drop="first"):
        if not cols:
            cols=self.col_cat

        for col in cols:
            self.onehotencoder_col(col, drop=drop)
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
    

if __name__ == "__main__":
    main()