import datetime
from tkinter import messagebox
import pandas as pd
import serial
from Data_processing import extract_gains
from Utils import debug
import time
import numpy as np
from tqdm import tqdm
from progress.bar import Bar
from alive_progress import alive_bar

epsilon = 1e-30

class Test:
    def __init__(self, test_type: str):
        self.type = test_type
        self.stop_test_flag = False
        self.stop_continuous = False
        self.steps = 0
        self.parameters = {
            "E1": -200,
            "E2": 500,
            "Ep": 1,
            "Gain": 4,
            "Rload": 10,
            "Rtia": 0
        }
        self.results = pd.DataFrame(
            columns=[
                "time",
                "raw_voltages",
                "raw_currents",
                "baseline",
                "normalized_gain",
                "peak_voltage",
                "peak_current",
                "half_heigths",
                "smooth_data",
                "concentration",
                "frequency"
            ])

    def update_param(self, params: dict) -> bool:
        try:
            self.parameters.update(params)
        except Exception:
            return debug()
        return True

    def get_params(self) -> dict:
        return self.parameters

    def get_index(self):
        return self.results.index

    def get_df(self) -> pd.DataFrame:
        return self.results

    def add_result(self, index: int, _time: float, _voltage: list[float], _current: list[float], frequency: float,
                   concentration: float = None) -> int:
        try:
            data = extract_gains(_voltage, _current)
            self.results.loc[index] = [
                _time,
                _voltage,
                _current,
                list(data["baseline coefs"]),
                list(data["gain coefs"]),
                data["peak voltage"],
                data["peak current"],
                data["half-height voltages"],
                list(data["smooth_data"]),
                concentration,
                frequency
            ]
            self.results.sort_index(axis=0, inplace=True)
            return 1
        except Exception:
            return debug()

    def run_test(self, comport, baudrate):
        pass

    def stop_test(self):
        self.stop_test_flag = True
        self.stop_continuous = True

    def get_steps(self):
        if self.parameters["E1"] < 0 and self.parameters["E2"] < 0:
            self.steps = abs(int(abs(self.parameters["E1"]) + self.parameters["E2"])) + 1
        elif self.parameters["E1"] > 0 and self.parameters["E2"] > 0:
            self.steps = abs(int(abs(self.parameters["E1"]) + self.parameters["E2"])) + 1
        else:
            self.steps = int(abs(self.parameters["E1"]) + abs(self.parameters["E2"])) + 1


class Titration(Test):
    def __init__(self):
        super(Titration, self).__init__("Titration")
        self.parameters.update({"Frequency": 100})
        self.parameters.update({"Amplitude": 25})
        self.parameters.update({"Concentration": 0})

    def run_test(self, comport, baudrate):
        self.stop_test_flag = False  # Reset the stop flag before running the test
        self.get_steps()
        try:
            dt = datetime.datetime.now()
            dt_float = float(dt.timestamp())
            dt = float(dt_float / 86400)
            ser = serial.Serial(port=comport, baudrate=baudrate)
            ser.read_all()
            data = "SWV,"

            if self.parameters["Concentration"] <= 0.0:
                return "Please enter a concentration bigger than zero"
            else:
                print(f"Titration data:")
                for param, value in self.parameters.items():
                    data = data + f"{param}:{value},"
                    print(f"{param}:{value},")

                self.get_steps()

                data = data[:-1]
                ser.write(data.encode())
                _time = []
                _voltage = []
                _current = []
                _index = self.results.shape[0]
                with alive_bar(self.steps) as bar:
                    while not self.stop_test_flag:
                        try:
                            if int(ser.inWaiting()) > 0:
                                line = ser.read(size=ser.inWaiting()).decode()
                                #print(line)
                                if line.find("Done") >= 0:
                                    break
                                elif line.find("time:") >= 0:
                                    lst = line.split(",")
                                    _time.append(float(lst[0].split(":")[1]))
                                    _voltage.append(float(lst[1].split(":")[1]))
                                    _current.append(float(lst[2].split(":")[1].strip()))
                                    bar()
                        except Exception as e:
                            return e
                if self.stop_test_flag:
                    self.stop_test_flag = False
                    return "Test stopped by user"
                return self.add_result(_index, dt, _voltage, _current, self.parameters["Frequency"],
                                       self.parameters["Concentration"])
        except Exception as e:
            return e


class CV(Test):
    def __init__(self):
        super(CV, self).__init__("CV")
        self.parameters.update({"Frequency": 100})
        self.parameters.update({"vertex1": 0})
        self.parameters.update({"vertex2": 200})
        self.parameters.update({"Cycles": 1})

    def run_test(self, comport, baudrate):
        ## do the experiment
        dt = datetime.datetime.now()
        dt = float(dt.timestamp() / 86400)
        ser = serial.Serial(port=comport, baudrate=baudrate)
        ser.read_all()
        data = "CV,"
        print(f"CV data:")
        for param, value in self.parameters.items():
            data = data + f"{param}:{value},"
            print(f"{param}:{value},")
        data = data[:-1]
        ser.write(data.encode())
        _time = []
        _voltage = []
        _current = []
        _index = self.results.shape[0]
        while 1:
            try:
                if int(ser.inWaiting()) > 0:
                    line = ser.read(size=ser.inWaiting()).decode()
                    if line.find("Done") >= 0:
                        break
                    elif line.find("experiment Starting") >= 0:
                        experiment_started = True
                    elif line.find("time:") >= 0:
                        lst = line.split(",")
                        _time.append(float(lst[0].split(":")[1]))
                        _voltage.append(float(lst[1].split(":")[1]))
                        _current.append(float(lst[2].split(":")[1].strip()))

            except Exception:
                return debug()
        return self.add_result(_index, dt, _voltage, _current, self.parameters["Frequency"])


class SWV(Test):
    def __init__(self):
        super(SWV, self).__init__("SWV")
        self.parameters.update({"Frequency": 100})
        self.parameters.update({"Amplitude": 25})
        self.parameters.update({"RunTime": 10})


    def run_test(self, comport, baudrate):
        ## do the experiment
        self.stop_test_flag = False  # Reset the stop flag before running the test
        dt = datetime.datetime.now()
        dt = float(dt.timestamp() / 86400)
        ser = serial.Serial(port=comport, baudrate=baudrate)
        ser.read_all()
        data = "SWV,"
        # print(f"SWV data:")
        for param, value in self.parameters.items():
            data = data + f"{param}:{value},"
        #     print(f"{param}:{value},")
        data = data[:-1]
        ser.write(data.encode())
        _time = []
        _voltage = []
        _current = []
        _index = self.results.shape[0]
        while not self.stop_test_flag:
            try:
                if int(ser.inWaiting()) > 0:
                    data = ser.read(size=ser.inWaiting()).decode()
                    lines = data.split('\n')

                    for line in lines:
                        line = line.strip()
                        if "Done" in line:
                            return self.add_result(_index, dt, _voltage, _current, self.parameters["Frequency"])
                        elif "time:" in line:
                            lst = line.split(",")
                            _time.append(float(lst[0].split(":")[1]))
                            _voltage.append(float(lst[1].split(":")[1]))
                            print("voltage: ", float(lst[1].split(":")[1]))
                            _current_value = lst[2].split(":")[1].strip()  # Remove unwanted characters
                            # Try to convert the cleaned string to a float
                            try:
                                _current.append(float(_current_value))
                            except ValueError:
                                print('VALUE ERROR')
                                print(_current_value)
                                print(_current_value.split("/"))
                        else:
                            break
            except Exception:
                return debug()
        if self.stop_test_flag:
            self.stop_test_flag = False
            return "Test stopped by user"
        return self.add_result(_index, dt, _voltage, _current, self.parameters["Frequency"])




if __name__ == "__main__":
    pass