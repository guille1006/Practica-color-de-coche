class Pyurification:

    def __init__(self, df, variable_dependiente):
        self.df = df
        self.uniques = df.nunique()
        self.len = len(df)


        if type(variable_dependiente)==str:
            self.col_independientes = self.variables_independientes()
            self.col_dependiente = variable_dependiente
        
        else:
            self.col_dependiente = self.variables_dependiente()

        self.col_numericas = self.variables_numericas()
        self.col_categoricas = self.variables_categoricas()
        self.col_dicotomicas = self.variables_dicotomicas()
        

