import pandas as pd


class ModelService:
    # pylint: disable=too-few-public-methods, invalid-name
    """
    A class for model service
    """

    def __init__(self, model, le):
        self.model = model
        self.le = le

    def predict(self, data):
        """
        Predict the data in the dictionary
        Args:
            dicts (dict): A dictionary that contains input features

        Returns:
            numpy.ndarray: Predicted data in the form of numpy array
        """
        print("-------------------------------\n\n")
        df = pd.DataFrame([data])
        print("-------------------------------\n\n")
        df["Sex"] = self.le.transform(df["Sex"])
        # numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # df[numerical_columns] = self.sr.transform(df[numerical_columns])

        y_pred = self.model.predict(df)

        return y_pred
