class Electrode_c:
    def __init__(self):
        # Define a private attribute with a leading underscore
        self._value = "None"

    def get(self):
        # Getter method
        return self._value

    def set(self, new_value):
        # Setter method
        if isinstance(new_value, str):
            self._value = new_value
        else:
            print("Invalid value. Please provide a string.")

class Experiment_c:
    def __init__(self):
        # Define a private attribute with a leading underscore
        self._value = "None"

    def get(self):
        # Getter method
        return self._value

    def set(self, new_value):
        # Setter method
        if isinstance(new_value, str):
            self._value = new_value
        else:
            print("Invalid value. Please provide a string.")

class Titration_c:
    def __init__(self):
        # Define a private attribute with a leading underscore
        self._value = "None"

    def get(self):
        # Getter method
        return self._value

    def set(self, new_value):
        # Setter method
        if isinstance(new_value, str):
            self._value = new_value
        else:
            print("Invalid value. Please provide a string.")

class Test_c:
    def __init__(self):
        # Define a private attribute with a leading underscore
        self._value = "None"

    def get(self):
        # Getter method
        return self._value

    def set(self, new_value):
        # Setter method
        if isinstance(new_value, str):
            self._value = new_value
        else:
            print("Invalid value. Please provide a string.")