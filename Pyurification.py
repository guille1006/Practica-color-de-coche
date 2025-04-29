class Pyurification:
    def __init__(self, df, v_depend=False, col_depend=False, col_cat=[]):
        self.df = df
        self.len = len(df)
        self.uniques = {}
        self.cols = df.columns

        if v_depend:
            self.v_depend = v_depend
            self.col_depend = None
            self.v_independ = df
            self.col_depend = df.columns

        elif col_depend:
            self.col_depend = col_depend
            self.v_depend = df[col_depend]
            self.col_independ = [x for x in df.columns if not col_depend]    #Posible error
            self.v_independ = df[self.col_independ]

        else:
            self.col_depend = None
            self.v_depend = None
            self.col_independ = df.columns
            self.v_independ = df

        self.col_dicotom = []
        self.col_cat = []
        self.col_num = []
#---------------------------------------------------------------------------------
    def select_type_variables(self, col_cat):
        col_to_analyze = [x for x in self.col_independ if not in col_cat]
       
        for col in col_to_analyze:
            serie = self.v_independ[col]
            self.uniques[col] = serie.nunique()

            if self.uniques[col] == 2:
                self.col_dicotom.append(col)
                self.df[col] = dicotom_to_number(df[col])

            elif self.uniques[col]>20:
                return
               



#---------------------------------------------------------------------------------
    def dicotom_to_number(self, series):
        val_unicos = series.unique()
        val_unicos.sort()
        return series.replace({val_unicos[0]:0, val_unicos[1]:1})


#---------------------------------------------------------------------------------
    def show_frecuencys(self, n_groups_max=250):
        resultados = {}
        frecuencias = []
       
        columns_showed = [x for x in self.columns if x<n_groups_max]

        for cols in columns_showed: