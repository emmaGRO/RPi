import asyncio
import cProfile
import importlib
import json
import math
import pstats
import struct
import tkinter.tix
import warnings
import time
import threading
import queue
import re

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import filedialog, messagebox, ttk
import bleak
import matplotlib
import nest_asyncio
from dateutil import tz
from matplotlib import dates, ticker, pyplot as plt
import BLE_connector_Bleak
from Process_CH_data import *
from Data_processing import *
import pickle
import datetime
import serial.tools.list_ports
from Tests import Test
from Plots import Plot
from Titrations import titration

from memory_profiler import profile

font1 = 'Helvetica 15 bold'
font2 = 'Helvetica 11 bold'
font3 = 'Helvetica 10'
# hotfix to run nested asyncio to correctly close Bleak without having to wait for timeout to reconnect to device again
nest_asyncio.apply()

matplotlib.use('TkAgg')  # Makes sure that all windows are rendered using tkinter

address_default = 'FE:B7:22:CC:BA:8D'
uuids_default = ['340a1b80-cf4b-11e1-ac36-0002a5d5c51b', ]
write_uuid = '330a1b80-cf4b-11e1-ac36-0002a5d5c51b'

sender_battery_voltage = "battery_voltage"
SUPPRESS_WARNINGS = True
if SUPPRESS_WARNINGS:
    warnings.filterwarnings("ignore")


class App(tk.Tk):
    """Main window of app based on tkinter framework.Runs asynchronously, dynamically scheduling which loop to run next depending on intervals."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        """:param loop: parent event loop for asynchronous execution, it is not unique in this app"""
        super().__init__()

        self.Titration_cBox = None
        self.new_data = {}
        self.loop = loop
        self.toggle_cursor = False
        self.width = 10

        self.path_list = []
        self.output_path = os.getcwd() + "\\output"
        self.data_path = os.getcwd() + "\\data"
        self.titration_path = "C:\\Users\\emmaf\\Documents\\Universite\\Stage T4\\RPi\\data_titration"
        self.create_directories([self.output_path, self.data_path])

        self.electrode_list = {}
        self.titration_list = {}
        self.current_electrode = None
        self.raw_data_df = None
        self.update_raw_data_graph = False
        self.titration_df = None
        self.update_titration_graph = False
        self.to_update_plots = False
        self.datapoint_select_N = 0
        self.thread_result = -1
        self.data_received = False
        self.isHill = False
        self.check_params = False
        self.continuous_running = False

        ################################################# Menu bar #######################################################


        def on_button_close():  # stop tasks before close the window.
            try:
                print('Exiting...')
                try:
                    self.loop.run_until_complete(self.stop_scanning_handle())
                    self.loop.run_until_complete(self.BLE_connector_instance.disconnect())

                except Exception:
                    pass
                for task in self.tasks:
                    self.tasks[task].cancel()
                self.loop.stop()
                self.destroy()
                self.quit()
                print('Exiting finished!')
            except Exception as e:
                print(e)
                debug()
                messagebox.showerror('Error', e.__str__())


        def on_button_Toggle_fit():
            self.isHill = not self.isHill
            self.update_titration_graph = True
            self.to_update_plots = True


        def keyPressed(event):
            try:
                if event.keysym == 'Left':
                    if not self.chkBtn_show_latest_voltammogram_var.get() and self.volta_slider.get() > 0:
                        self.volta_slider.set(self.volta_slider.get() - 1)
                elif event.keysym == 'Right':
                    if not self.chkBtn_show_latest_voltammogram_var.get() and self.volta_slider.get() < len(
                            self.titration_df):
                        self.volta_slider.set(self.volta_slider.get() + 1)
            except Exception:
                pass

        self.bind_all('<Key>', keyPressed)

        menubar = tk.Menu(self)

        Graphmenu = tk.Menu(menubar, tearoff=0)
        Graphmenu.add_command(label="Toggle Hill/Linear fit", command=on_button_Toggle_fit)
        menubar.add_cascade(label="Graph", menu=Graphmenu)


        self.config(menu=menubar)

        self.tasks = {}  # list of tasks to be continuously executed at the same time (asynchronously, not in parallel)

        ###############################################################################################################
        self.protocol("WM_DELETE_WINDOW", on_button_close)  # the red x button
        self.wm_title("SwiftMote")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height - 100}+1-43")  # the size of the GUI
        self.resizable(True, True)
        # initialize the right part of the GUI
        self.frameGraph = tk.Frame(master=self, highlightbackground="black", highlightthickness=1)  # div
        self.plots = Plot(master=self, frame=self.frameGraph)

        # Initialize the left part of the GUI
        self.frameControls = tk.Frame(master=self, highlightbackground="black", highlightthickness=1)  # div
        self.init_controls(master=self.frameControls)

        self.frameControls.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.frameGraph.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.tasks["UI"] = loop.create_task(self.update_ui_loop(interval=1 / 60), name="UI")
        time.sleep(0.005)  # small delay to let dicts init
        self.tasks["Plot"] = loop.create_task(self.update_plot_loop(interval=1 / 60), name="Plot")
        # self.tasks["Autosave"] = loop.create_task(self.autosave_loop(percentage_of_time=20), name="Autosave")
        #################################################################
        # Testing purposes
        self.time_type = True
        #################################################################

    def create_directories(self, dir_list):
        for path in dir_list:
            if not os.path.exists(path):
                os.makedirs(path)

    def init_controls(self, master):

        frameTestVariables = tk.Frame(master=master, highlightbackground="black", highlightthickness=1)  # div

        """Initializes controls param master: reference to parent object """
        self.current_values = {}  # values from variable fields that can be sent to the device
        self.option_cbox_values = {}
        self.graph_values = {}
        self.calib_data_dict = {}

        ############################ Serial Device selection #############################################
        #
        # def refresh_Serial_devices():
        #     try:
        #         serial_devices_list = serial.tools.list_ports.comports()
        #         ports = []
        #         for port, _, _ in sorted(serial_devices_list):
        #             ports.append(port)
        #
        #         if ports:
        #             self.comport_cbox["state"] = "readonly"
        #             self.comport_cbox['values'] = ports
        #     except Exception as e:
        #         self.print(e)
        #         debug()
        #         messagebox.showerror('Error', 'No serial devices connected to scan for devices\n\n' + e.__str__())
        #
        # def apply_selected_Serial_device(event):
        #     pass
        #     self.tasks["Serial"] = loop.create_task(self.register_data_callbacks_serial(interval=5), name="Serial")


        ############################ Frame init for data control #############################################

        frameElectrode = tk.Frame(master=master,
                                  highlightbackground="black",
                                  highlightthickness=1,
                                  width=self.winfo_width()
                                  )  # div
        tk.Label(master=frameElectrode, text="Electrode", font=font1).pack(side=tk.TOP)
        frameElectrodebox = tk.Frame(master=frameElectrode,
                                     width=40)
        frameElectrodebox.columnconfigure(0, weight=1)

        frameExperiment = tk.Frame(master=frameElectrode,
                                   highlightbackground="black",
                                   highlightthickness=0)  # div
        tk.Label(master=frameExperiment, text="Experiment", font=font2).pack(side=tk.TOP)

        frameTitration = tk.Frame(master=frameExperiment,
                                   highlightbackground="black",
                                   highlightthickness=0)  # div
        tk.Label(master=frameTitration, text="Load Titration", font=font2).pack(side=tk.TOP)

        frameExpType = tk.Frame(master=frameExperiment,
                                highlightbackground="black",
                                highlightthickness=0)  # div
        tk.Label(master=frameExpType, text="Displayed data", font=font2).pack(side=tk.TOP)

        frameTest_results = tk.Frame(master=frameExpType,
                                     highlightbackground="black",
                                     highlightthickness=0)  # div

        frameTest_params = tk.Frame(master=master,
                                    highlightbackground="black",
                                    highlightthickness=1
                                    )  # div

        frameTest_params_params = tk.Frame(master=frameTest_params,
                                           highlightbackground="black",
                                           highlightthickness=1
                                           )  # div
        tk.Label(master=frameTest_params_params, text="Test parameters", font=font1).pack(side=tk.TOP)
        frameTest_params_btn = tk.Frame(master=frameTest_params,
                                        highlightbackground="black",
                                        highlightthickness=1
                                        )  # div

        frameTest_params_btn.columnconfigure(0, weight=1)
        frameTest_params_btn.columnconfigure(1, weight=1)
        frameTest_params_btn.columnconfigure(2, weight=1)

        ############################ Electrode selection with dropdown Experiment updating #############################################
        def update_electrode_list():
            elec_list = list(os.listdir(f"{self.data_path}"))
            self.Electrode_cBox["values"] = elec_list

        self.Electrode_cBox = ttk.Combobox(master=frameElectrodebox,
                                           values=[],
                                           width=self.width,
                                           justify="center",
                                           font=font3,
                                           postcommand=update_electrode_list)

        self.Electrode_cBox.bind('<<ComboboxSelected>>', lambda event: set_electrode(event))
        self.Electrode_cBox.pack(side=tk.TOP, fill=tk.X)

        add_electrode_btn = tk.Button(master=frameElectrodebox, state="normal", text="Add new electrode",
                                      command=lambda: add_electrode())
        add_electrode_btn.pack(side=tk.TOP, fill=tk.X)
        del_electrode_btn = tk.Button(master=frameElectrodebox, state="normal", text="Delete electrode",
                                      command=lambda: del_electrode())
        del_electrode_btn.pack(side=tk.TOP, fill=tk.X)
        frameElectrodebox.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        def del_electrode():
            electrode = self.current_electrode
            electrode.delete(self.data_path)
            self.electrode_list.pop(electrode.name)
            self.print(f"{electrode.name} deleted successfully")
            self.Electrode_cBox.set("")

        def load_electrode(name):
            with open(f"{self.data_path}\\{name}", "rb") as f:
                self.electrode_list[name] = pickle.load(f)

        def load_titration(name):
            self.update_titration_graph = True
            with open(f"{self.titration_path}\\{name}", "rb") as f:
                self.titration_list[name] = pickle.load(f)

        def set_electrode(event):
            electrode_name = self.Electrode_cBox.get()
            # if electrode_name not in self.electrode_list.keys():
            #     load_electrode(electrode_name)
            load_electrode(electrode_name)
            self.current_electrode = self.electrode_list[electrode_name]
            self.Experiment_cBox['state'] = "active"
            self.Experiment_cBox.set("")
            self.titration_df = None
            self.test_cBox.set("")
            self.latest_volta_btn["state"] = "disabled"
            self.raw_data_df = None
            self.to_update_plots = True
            add_experiment_btn['state'] = "active"
            del_experiment_btn['state'] = "active"
            save_titration_btn['state'] = "active"

        def set_new_electrode():
            electrode_name = self.Electrode_cBox.get()
            load_electrode(electrode_name)
            self.current_electrode = self.electrode_list[electrode_name]
            self.Experiment_cBox['state'] = "active"
            self.Experiment_cBox.set("")
            self.titration_df = None
            self.test_cBox.set("")
            self.latest_volta_btn["state"] = "disabled"
            self.raw_data_df = None
            self.to_update_plots = True
            add_experiment_btn['state'] = "active"
            del_experiment_btn['state'] = "active"

        def add_electrode():
            name = self.Electrode_cBox.get()
            if os.path.isfile(f"{self.data_path}\\{name}") or name in self.electrode_list.keys():
                messagebox.showerror('Error', f'{name} already exist, please modify name')
            elif name == "":
                messagebox.showerror('Error', f'please add electrode name')
            else:
                self.electrode_list[name] = Electrode(name)
                self.electrode_list[name].save(self.data_path)
                self.Electrode_cBox.set(self.electrode_list[name].name)
                set_new_electrode()

        def start_electrode():
            base_name = "electrode"
            number = 0
            while os.path.isfile(
                    f"{self.data_path}\\{base_name}_{number}"):
                number += 1
            name = f"{base_name}_{number}"
            self.Electrode_cBox.set(name)
            add_electrode()
            electrode_name = self.Electrode_cBox.get()
            # if electrode_name not in self.electrode_list.keys():
            #     load_electrode(electrode_name)
            load_electrode(electrode_name)
            self.current_electrode = self.electrode_list[electrode_name]
            self.Experiment_cBox['state'] = "active"
            self.Experiment_cBox.set("")
            self.titration_df = None
            self.test_cBox.set("")
            self.latest_volta_btn["state"] = "disabled"
            self.raw_data_df = None
            self.to_update_plots = True
            add_experiment_btn['state'] = "active"
            del_experiment_btn['state'] = "active"
            save_titration_btn['state'] = "active"


        # ############################# Experiment selection from selected Electrode ######################################################
        def update_experience_list():
            self.Experiment_cBox['values'] = self.current_electrode.get_experiments()

        self.Experiment_cBox = tk.ttk.Combobox(master=frameExperiment,
                                               width=40,
                                               state="disabled",
                                               postcommand=update_experience_list)

        self.Experiment_cBox.bind('<<ComboboxSelected>>', lambda event: set_experiment(event))
        self.Experiment_cBox.pack(side=tk.TOP, fill=tk.X)

        add_experiment_btn = tk.Button(master=frameExperiment, text="Add Experiment", state="disabled",
                                       command=lambda: add_experiment())
        add_experiment_btn.pack(side=tk.TOP, fill=tk.X)
        del_experiment_btn = tk.Button(master=frameExperiment, text="Delete Experiment", state="disabled",
                                       command=lambda: del_experiment())
        del_experiment_btn.pack(side=tk.TOP, fill=tk.X)
        save_titration_btn = tk.Button(master=frameExperiment, text="Save Titration", state="disabled",
                                       command=lambda: save_titration())
        save_titration_btn.pack(side=tk.TOP, fill=tk.X)



        # ############################# Titration selection from file ######################################################
        def update_titration_list():
            titr_list = list(os.listdir(f"{self.titration_path}"))
            self.Titration_cBox['values'] = titr_list

        def set_titration(event):
            file_name = self.Titration_cBox.get()
            match = re.match(r"^(.*)\((.*)\)\.csv$", file_name)
            if match:
                electrode_name = match.group(1)
                titration_name = match.group(2)
            else:
                titration_name = file_name
            self.titration_df = None
            if file_name:
                load_titration(titration_name)
                self.current_titration = self.titration_list[file_name]
                self.titration_df = self.current_titration.get_df().sort_values(by=["concentration"])
                if self.plots.prev_min_pt is None:
                    self.plots.min_pt = list(self.titration_df["concentration"])[0]
                    self.plots.max_pt = list(self.titration_df["concentration"])[-1]
                # self.test_cBox.set("")
                # self.latest_volta_btn["state"] = "disabled"
                # self.raw_data_df = None
                self.to_update_plots = True
            pass

        def force_set_titration():
            self.Titration_cBox.set("test")
            file_name = self.Titration_cBox.get()
            match = re.match(r"^(.*)\((.*)\)\.csv$", file_name)
            if match:
                electrode_name = match.group(1)
                titration_name = match.group(2)
            else:
                titration_name = file_name
            self.titration_df = None
            if file_name:
                load_titration(titration_name)
                self.current_titration = self.titration_list[file_name]
                self.titration_df = self.current_titration.get_df().sort_values(by=["concentration"])
                if self.plots.prev_min_pt is None:
                    self.plots.min_pt = list(self.titration_df["concentration"])[0]
                    self.plots.max_pt = list(self.titration_df["concentration"])[-1]
                # self.test_cBox.set("")
                # self.latest_volta_btn["state"] = "disabled"
                # self.raw_data_df = None
                self.to_update_plots = True
            pass

        self.Titration_cBox = tk.ttk.Combobox(master=frameTitration,
                                               width=40,
                                               state="disabled",
                                               postcommand=update_titration_list)

        self.Titration_cBox.bind('<<ComboboxSelected>>', lambda event: set_titration(event))
        self.Titration_cBox.pack(side=tk.TOP, fill=tk.X)
        frameTitration.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        def add_experiment():
            if self.Experiment_cBox.get() == "":
                messagebox.showerror('Error', f'please add experiment name')
            else:
                name = self.Experiment_cBox.get()
                electrode = self.current_electrode
                if name in electrode.get_experiments():
                    messagebox.showerror('Error', f'{name} already exist, please modify name')
                else:
                    electrode.create_experiment(self.Experiment_cBox.get())
                    self.print(f"{name} created successfully")
                    self.Experiment_cBox.event_generate('<<ComboboxSelected>>')
                    electrode.save(self.data_path)

        def del_experiment():
            name = self.Experiment_cBox.get()
            electrode = self.current_electrode
            if name in electrode.get_experiments():
                electrode.del_experiment(name)
                electrode.save(self.data_path)
                self.print(f"{name} deleted successfuly")
                self.Experiment_cBox.set("")
                self.titration_df = None
                block_tests()
            else:
                messagebox.showerror('Error', f"{name} doesn't exists in {electrode.name}")

        def save_titration():
            if self.Experiment_cBox.get() == "":
                messagebox.showerror('Error', f'please select an experiment')
            else:
                exper_name = self.Experiment_cBox.get()
                electrode_name = self.Electrode_cBox.get()
                name = f"{electrode_name}({exper_name}).csv"
                folder = str((f"{self.titration_path}"))
                # Create titration
                titr = titration(electrode_name, exper_name, folder, self.titration_df)
                self.titration_list[name] = titr
                titr.save(folder)

        def set_experiment(event):
            self.titration_df = None
            experiment_name = self.Experiment_cBox.get()
            if experiment_name not in self.current_electrode.get_experiments():
                self.print(f"{experiment_name} doesn't exist")
            else:
                self.test_cBox['state'] = 'active'
                create_titration_btn["state"] = 'active'
                create_Lovric_btn["state"] = 'active'
                create_Volta_btn["state"] = 'active'
                self.test_cBox.event_generate('<<ComboboxSelected>>')
                if self.current_electrode.get_tests(experiment_name)["Titration"].get_df().shape[0] > 0:
                    self.titration_df = self.current_electrode.get_tests(experiment_name)["Titration"].get_df().sort_values(by=["concentration"])
                    self.plots.min_pt = list(self.titration_df["concentration"])[0]
                    self.plots.max_pt = list(self.titration_df["concentration"])[-1]
                    self.update_titration_graph = True
                    self.Titration_cBox['state'] = 'disabled'
                else:
                    self.Titration_cBox['state'] = 'active'
            self.to_update_plots = True

        def start_experiment():
            self.titration_df = None
            base_name = "experiment"
            number = 0
            while os.path.isfile(
                    f"{self.data_path}\\{base_name}_{number}"):
                number += 1
            name = f"{base_name}_{number}"
            self.Experiment_cBox.set(name)
            experiment_name = self.Experiment_cBox.get()
            add_experiment()
            if experiment_name not in self.current_electrode.get_experiments():
                self.print(f"{experiment_name} doesn't exist")
            else:
                self.test_cBox['state'] = 'active'
                create_titration_btn["state"] = 'active'
                create_Lovric_btn["state"] = 'active'
                create_Volta_btn["state"] = 'active'
                self.test_cBox.event_generate('<<ComboboxSelected>>')
                if self.current_electrode.get_tests(experiment_name)["Titration"].get_df().shape[0] > 0:
                    self.titration_df = self.current_electrode.get_tests(experiment_name)["Titration"].get_df().sort_values(by=["concentration"])
                    self.plots.min_pt = list(self.titration_df["concentration"])[0]
                    self.plots.max_pt = list(self.titration_df["concentration"])[-1]
                    self.update_titration_graph = True
                    self.Titration_cBox['state'] = 'disabled'
                else:
                    self.Titration_cBox['state'] = 'active'
            self.to_update_plots = True

        frameExperiment.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        # ############################# Experiment type selection from selected Experiment ######################################################
        def block_tests():
            self.test_cBox.set("")
            self.test_cBox['state'] = 'disabled'
            create_titration_btn["state"] = 'disabled'
            create_Lovric_btn["state"] = 'disabled'
            create_Volta_btn["state"] = 'disabled'
            for child in frameTest_params_params.winfo_children():
                child.destroy()
            update_plot()
            self.raw_data_df = None

        def update_test_list():
            self.test_cBox["values"] = list(self.current_electrode.get_tests(self.Experiment_cBox.get()).keys())
            self.latest_volta_btn["state"] = "active"

        self.test_cBox = tk.ttk.Combobox(master=frameExpType,
                                         width=40,
                                         state="disabled",
                                         postcommand=update_test_list)

        self.test_cBox.bind('<<ComboboxSelected>>', lambda event: set_test_graph(event))
        self.test_cBox.pack(side=tk.TOP, fill=tk.X)
        frameExpType.pack(side=tk.TOP, fill=tk.BOTH, expand=False)



        def set_test_graph(event):
            if self.test_cBox.get() != "":
                self.raw_data_df = self.current_electrode.get_tests(self.Experiment_cBox.get())[
                    self.test_cBox.get()].get_df()
                self.plots.rt_concentration_data["rt concentration"].set_data([], [])
                self.update_raw_data_graph = True
                self.to_update_plots = True

        ############################################### Results ##################################################
        frameTestVariablesGrid = tk.Frame(master=frameTest_results)
        frameTestVariablesGrid.pack(side=tk.TOP, anchor=tk.N)  # div 2

        tk.Label(master=frameExpType, text="Results", font=font2).pack(side=tk.TOP)

        def update_plot():
            self.to_update_plots = True

        self.chkBtn_show_latest_voltammogram_var = tk.IntVar(value=0)
        self.latest_volta_btn = tk.Checkbutton(master=frameExpType,
                                               text="Latest voltammogram",
                                               variable=self.chkBtn_show_latest_voltammogram_var,
                                               command=update_plot,
                                               font=font3,
                                               state="disabled")

        self.latest_volta_btn.pack(side=tk.TOP)

        def refresh_slider(event):
            if self.test_cBox.get() != "":
                update_raw_data_graph(event)

        self.volta_slider_val = tk.IntVar(value=1)
        self.volta_slider = tk.Scale(frameExpType,
                                     orient='horizontal',
                                     showvalue=False,
                                     sliderlength=0,
                                     length=325,
                                     command=refresh_slider,
                                     from_=1,
                                     to=1,
                                     state="disabled"
                                     )
        self.volta_slider_text = tk.Label(master=frameExpType, font=font3)
        self.volta_slider_text.pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=0, pady=0)
        self.volta_slider.pack(side=tk.TOP, fill=tk.Y, expand=1, padx=0, pady=0)

        def update_raw_data_graph(event):
            self.update_raw_data_graph = True
            self.to_update_plots = True

        def set_gain(event):
            try:
                value = self.gain_Cbox.get()
                if int(float(value)) == 0:
                    self.print(f"Gain is zero {self.gain_Cbox.get()}")
                    self.Rtia_Ebox["state"] = "normal"
                else:
                    self.print(f"Gain is not zero {self.gain_Cbox.get()}")
                    self.Rtia_Ebox["state"] = "normal"
                    self.Rtia_Ebox.delete(0, "end")
                    self.Rtia_Ebox.insert(0, 0)
                    self.Rtia_Ebox["state"] = "disabled"
            except Exception:
                debug()

        def change_params(test: Test):
            pass

        def Update_test_variable_frame(test: Test):  # used to set variable list for the test chosen in the combo box self.tests_cbox
            for child in frameTest_params_params.winfo_children():
                child.destroy()
            tk.Label(master=frameTest_params_params, text=test.type, font=font2).pack(side=tk.TOP)
            frameTestVariablesGrid = tk.Frame(master=frameTest_params_params)
            frameTestVariablesGrid.pack(side=tk.TOP, anchor=tk.N)  # div 2
            index = 0
            self.test_params = {}
            for variable, value in test.get_params().items():
                self.test_params[variable] = tk.DoubleVar(value=value)
                tk.Label(master=frameTestVariablesGrid, text=variable, font=font3).grid(row=index, column=0, sticky='W')
                if variable != "Rload" and variable != "Gain" and variable != "Rtia" and variable != "RunTime":
                    Entry_box = tk.Entry(master=frameTestVariablesGrid,
                                         textvariable=self.test_params[variable],
                                         width=self.width + 1,
                                         font=font3,
                                         justify="center")
                    Entry_box.grid(row=index, column=1, pady=1)
                else:
                    if variable == "Rload":
                        Combo_box = tk.ttk.Combobox(master=frameTestVariablesGrid,
                                                    textvariable=self.test_params[variable],
                                                    width=self.width - 2,
                                                    font=font3,
                                                    values=["10", "33", "50", "100"],
                                                    justify="center",
                                                    state="readonly")
                        Combo_box.grid(row=index, column=1, pady=1)
                    elif variable == "Gain":
                        self.gain_Cbox = tk.ttk.Combobox(master=frameTestVariablesGrid,
                                                         textvariable=self.test_params[variable],
                                                         width=self.width - 2,
                                                         font=font3,
                                                         values=["0", "1", "2", "3", "4", "5", "6", "7"],
                                                         justify="center",
                                                         state="readonly")
                        self.gain_Cbox.bind('<<ComboboxSelected>>', lambda event: set_gain(event))
                        self.gain_Cbox.grid(row=index, column=1, pady=1)
                    elif variable == "Rtia":
                        self.Rtia_Ebox = tk.Entry(master=frameTestVariablesGrid,
                                                  textvariable=self.test_params[variable],
                                                  width=self.width + 1,
                                                  font=font3,
                                                  justify="center",
                                                  state="disabled")
                        self.Rtia_Ebox.grid(row=index, column=1, pady=1)
                        self.gain_Cbox.event_generate('<<ComboboxSelected>>')

                index += 1
            if test.type == 'SWV':
                self.chkBtn_change_param = tk.IntVar(value=0)
                change_param_btn = tk.Checkbutton(master=frameTestVariablesGrid,
                                                  text="Change parameters",
                                                  variable=self.chkBtn_change_param,
                                                  command=lambda: change_params(
                                                      self.current_electrode.get_tests(self.Experiment_cBox.get())[
                                                          test.type]),
                                                  font=font3,
                                                  state="normal")
                change_param_btn.grid(row=index, columnspan=2, column=0, sticky="nesw")
                time_box = tk.Entry(master=frameTestVariablesGrid,
                                    textvariable=self.test_params["RunTime"],
                                    width=self.width + 1,
                                    font=font3,
                                    justify="center",
                                    state="normal")
                time_box.grid(row=index - 1, column=1, pady=1)
                Run_test_btn = tk.Button(master=frameTestVariablesGrid, state="active", text="Run Test",
                                         command=lambda: run_test(test))
                Run_test_btn.grid(row=index+1, column=0, sticky="nesw")
                Continuous_Run_test_btn = tk.Button(master=frameTestVariablesGrid, state="active",
                                                    text="Continuous Run", command=lambda: run_continuous_test(test))
                Continuous_Run_test_btn.grid(row=index + 1, column=1, sticky="nesw")
                Stop_test_btn = tk.Button(master=frameTestVariablesGrid, state="active", text="Stop Test",
                                          command=test.stop_test)
                Stop_test_btn.grid(row=index + 1, column=2, sticky="nesw")
            else:
                Run_test_btn = tk.Button(master=frameTestVariablesGrid, state="active", text="Run Test",
                                         command=lambda: run_test(test))
                Run_test_btn.grid(row=index, columnspan=2, sticky="nesw")
                Stop_test_btn = tk.Button(master=frameTestVariablesGrid, text="Stop Test", command=test.stop_test)
                Stop_test_btn.grid(row=index + 1, columnspan=2, sticky="nesw")

        create_titration_btn = tk.Button(master=frameTest_params_btn, state="disabled", text="Create Titration",
                                         command=lambda: Create_titration())
        create_Lovric_btn = tk.Button(master=frameTest_params_btn, state="disabled", text="Create CV",
                                      command=lambda: Create_CV())
        create_Volta_btn = tk.Button(master=frameTest_params_btn, state="disabled", text="Create SWV",
                                     command=lambda: Create_SWV())
        create_titration_btn.grid(row=0, column=0, sticky="nesw")
        create_Lovric_btn.grid(row=0, column=1, sticky="nesw")
        create_Volta_btn.grid(row=0, column=2, sticky="nesw")

        def Create_titration():
            Update_test_variable_frame(self.current_electrode.get_tests(self.Experiment_cBox.get())["Titration"])

        def Create_CV():
            Update_test_variable_frame(self.current_electrode.get_tests(self.Experiment_cBox.get())["CV"])

        def Create_SWV():
            Update_test_variable_frame(self.current_electrode.get_tests(self.Experiment_cBox.get())["SWV"])
        def Force_Create_SWV():
            Update_test_variable_frame(self.current_electrode.get_tests(self.Experiment_cBox.get())["SWV"])

        def run_test_thread(test, comport):
            self.thread_result = test.run_test(comport, 115200)

        def handle_test_results_delayed(test):
            while self.thread_result == -1:
                app.update()  # Allow the GUI to update
                time.sleep(0.1)  # Sleep for 100ms
            handle_test_results(test)  # Call handle_test_results once result is not -1
            self.data_received = True

        def handle_test_results(test):
            if self.thread_result == 1:
                self.current_electrode.save(self.data_path)
                self.print("Test ran successfully")
                app.update()
            elif not self.continuous_running:
                messagebox.showerror('Error 2', self.thread_result.__str__())
            else:
                if self.thread_result == "Test stopped by user":
                    messagebox.showerror('Error', self.thread_result.__str__())
                else:
                    print('Error', self.thread_result.__str__())
            self.thread_result = -1
            self.Experiment_cBox.event_generate('<<ComboboxSelected>>')
            self.Titration_cBox.event_generate('<<ComboboxSelected>>')
            self.test_cBox.event_generate('<<ComboboxSelected>>')
            self.volta_slider.set(len(test.get_df()) + 1)

        def run_test(test: Test):
            self.test_runned = True
            try:
                self.thread_result == -1
                self.data_received = False
                param = dict([(p[0], p[1].get()) for p in self.test_params.items()])
                test.update_param(param)
                ## ajout comm8
                comport = "COM8"
                threading.Thread(target=run_test_thread, args=(test, comport)).start()
                handle_test_results_delayed(test)
            except Exception as e:
                messagebox.showerror('Error 1', e.__str__())

        def run_continuous_test(test:Test):
            test.stop_continuous = False
            self.continuous_running = True

            def update_gui():
                app.update()

            def run_test_and_update_gui():
                try:
                    run_test(test)
                    update_gui()
                    index = 0
                    if index < test.get_params()["RunTime"] and not test.stop_continuous:
                        while self.data_received == False:
                            pass
                        app.after(3000, run_test_and_update_gui)  # Schedule the next run
                    else:
                        self.continuous_running = False
                except Exception as e:
                    messagebox.showerror('Error 1', e.__str__())

            run_test_and_update_gui()  # Start the continuous test

        frameTest_params_params.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        frameTest_params_btn.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        ###Ajout
        start_electrode()
        ######################################## Info screen params ########################################################################
        def on_button_clear_info():
            self.info_screen.config(state='normal')
            self.info_screen.delete(1.0, 'end')
            self.info_screen.config(state='disabled')

        frameControlsInfo = tk.Frame(master=master,
                                     highlightbackground="black",
                                     highlightthickness=1,
                                     )

        self.clear_btn = tk.Button(master=frameControlsInfo, text="Clear Info screen", command=on_button_clear_info)
        self.clear_btn.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

        frameControlsInfo_grid = tk.Frame(master=frameControlsInfo,
                                          highlightbackground="black",
                                          )  # div
        frameControlsInfo_grid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scroll_bar = tk.Scrollbar(master=frameControlsInfo_grid)
        self.scroll_bar.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.info_screen = tk.Text(master=frameControlsInfo_grid,
                                   state="normal",
                                   width=40,
                                   yscrollcommand=self.scroll_bar.set,
                                   wrap='word',
                                   )
        self.scroll_bar.config(command=self.info_screen.yview)
        self.info_screen.pack(side=tk.LEFT, fill=tk.BOTH)

        ################################################################################################################
        frameElectrode.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        frameTest_params.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        frameTestVariables.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        frameControlsInfo.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ################################################################################################################
        ###AJOUT
        start_experiment()
        force_set_titration()
        Force_Create_SWV()
        test = self.current_electrode.get_tests(self.Experiment_cBox.get())["SWV"]
        time.sleep(5)
        run_test(test)


    async def register_data_callbacks_bleak(self):
        """Sets up notifications using Bleak, and attaches callbacks"""
        # initialize time variables
        self.last_transaction_time = datetime.datetime(1999, 1, 1, 0, 0, 0, 0,
                                                       tzinfo=datetime.timezone.utc)  # to trigger offset adjustment
        self.offest_time = datetime.timedelta(days=0, seconds=0, microseconds=0)
        self.time_changed_threshold = 0

        # create an object, but not connect yet
        self.BLE_connector_instance = BLE_connector_Bleak.BLE_connector(to_connect=False)
        await self.start_scanning_process()
        await self.BLE_connector_instance.keep_connections_to_device(uuids=uuids_default,
                                                                     callbacks=[self.on_new_data_callback_SWV, ])

    async def register_data_callbacks_serial(self, interval):
        try:
            waiter = StableWaiter(interval=interval)
        except Exception as e:
            print(e)
            return
        while True:
            try:
                while self.new_data == {}:
                    await waiter.wait_async()
                    continue
                data = [self.new_data["voltages"], self.new_data["currents"], self.new_data["frequency"]]
                self.new_data = {}
                for i in range(len(data[0])):
                    self.print(str(data[0][i]) + ', ' + str(data[1][i]))
                await self.on_new_data_callback_SWV(sender=self.sender_SWV, data_joined=data)

            except Exception:
                debug()

    async def on_new_data_callback_SWV(self, sender, data: bytearray = bytearray(b'\x01\x02\x03\x04'),
                                       data_joined=None):
        try:
            time_delivered = datetime.datetime.now(datetime.timezone.utc)
            if data_joined:
                data_joined = data_joined
                time_best_effort = time_delivered
            else:
                status, transaction = self.process_packet(data=data, time_delivered=time_delivered)
                if status != 0:  # Transaction is not complete
                    return

                data_joined = transaction.get_joined_data()
                if data_joined is None:
                    return

                    # Time can only increment. If it decremented, it likely means BlueNRG chip rebooted.
                if self.last_transaction_time <= transaction.get_min_time_of_transaction_creation():
                    # Time incremented or stayed the same
                    pass
                else:

                    self.time_changed_threshold += 1
                    if self.time_changed_threshold > 1:
                        self.time_changed_threshold = 0

                        # Calculate new offset
                        self.offest_time = transaction.get_min_time_of_transaction_delivery() - transaction.get_min_time_of_transaction_creation()
                        temp = ['Time decremented, offset adjusted', self.offest_time]
                        self.print(temp)
                    else:
                        self.print("Likely stale data, discarding Transaction")
                        return
                self.last_transaction_time = transaction.get_min_time_of_transaction_creation()

                time_best_effort = transaction.get_min_time_of_transaction_creation() + self.offest_time

            if sender not in self.SwiftMote_df.get_df_list():  # if data recieved from this sender very first time, create new Dataframe
                self.SwiftMote_df.add_dataframe(sender)

            self.SwiftMote_df.add_data_to_df(sender,
                                             time_best_effort.timestamp() / 86400,
                                             # days since 1970-01-01 in UTC, this is required for matplotlib
                                             time_best_effort.astimezone().strftime("%Y-%m-%d %H:%M:%S:%f"),
                                             # "%Y-%m-%d %H:%M:%S:%f %z"
                                             data_joined[0],
                                             data_joined[1],
                                             data_joined[2]
                                             )
            self.to_update_plots = True

        except ValueError as e:
            self.print(e)
            debug()
        except Exception as e:
            self.print(e)
            debug()
            messagebox.showerror('Error', e.__str__())

    def process_packet(self, data, time_delivered):
        """Processes a packet and returns transaction when finalized
        Worst part of the code, needs to be optimized, but I don't know how"""
        transaction_finished = None

        if "transaction_in_progress" not in dir(self):  # if not defined, create new transaction
            self.transaction_in_progress = Transaction()  # Transaction object, called only when the app starts
        #  -1: transaction is complete, -2: transaction was just completed because new transaction code was detected
        if self.transaction_in_progress.add_packet(data=data, time_delivered=time_delivered) in [-1, -2]:
            # if error, maybe it is beginning of a new transaction? Try to add packet second time
            transaction_finished = self.transaction_in_progress  # save reference of the previous transaction that has completed
            self.transaction_in_progress = Transaction()  # create a new Transaction object, used in the following transactions
            # -2 should never happen
            if self.transaction_in_progress.add_packet(data=data, time_delivered=time_delivered) == -1:
                # self.print("Error of starting new transaction, datahex: ", datahex)
                return -1, None

        if transaction_finished is not None and transaction_finished.finalized:
            return 0, transaction_finished
        if self.transaction_in_progress.finalized:  # Will execute only if 1 packet was expected, in theory
            return 0, self.transaction_in_progress
        else:
            # self.print("Transaction is not complete yet")
            return -2, None

    async def update_plot_loop(self, interval):
        """Updates plots inside UI, at regular intervals

        param interval: minimum time between 2 updates, time of execution is taken in account
        """
        self.print('Plotting started')
        waiter = StableWaiter(interval=interval)
        while True:
            try:
                await waiter.wait_async()

                if self.to_update_plots is False:
                    # optimization to prevent re-drawing when there is no new data or when plotting is paused
                    continue
                self.to_update_plots = False
                ######################################## Titration Graph ###################################################
                if self.titration_df is not None:
                    if self.plots.prev_min_pt is not None:
                        Plot.prev_min_pt = self.plots.min_pt
                    if self.plots.prev_max_pt is not None:
                        self.plots.max_pt = Plot.prev_max_pt

                    if self.update_titration_graph:
                        concentration = list(self.titration_df['concentration'])
                        max_gain = []
                        for i in range(len(self.titration_df['raw_voltages'].iloc[:])):
                            g = self.titration_df['peak_current'].iloc[i]
                            max_gain.append(g)
                        # normalized
                        first_peak_value = max_gain[0]
                        max_gain = [x / first_peak_value for x in max_gain]
                        max_gain = [(x-1)*100 for x in max_gain]
                        if self.isHill:
                            if concentration[concentration.index(self.plots.min_pt)] < concentration[
                                concentration.index(self.plots.max_pt)]:
                                self.hf = HillFit(concentration[
                                                  concentration.index(self.plots.min_pt):concentration.index(
                                                      self.plots.max_pt) + 1],
                                                  max_gain[concentration.index(self.plots.min_pt):concentration.index(
                                                      self.plots.max_pt) + 1])
                                self.hf.fitting()
                            else:
                                conc = concentration[concentration.index(self.plots.min_pt):concentration.index(
                                    self.plots.max_pt) + 1]
                                gain = max_gain[concentration.index(self.plots.min_pt):concentration.index(
                                    self.plots.max_pt) + 1]
                                gain.reverse()
                                self.hf = HillFit(conc, gain)
                                self.hf.fitting()
                                self.hf.y_fit = np.flip(self.hf.y_fit)

                            self.plots.titration_data["titration"].set_data(concentration, max_gain)
                            self.plots.titration_data["fit"].set_data(self.hf.x_fit, self.hf.y_fit)
                            self.plots.titration_data["fit"].set_label(
                                f"$R^2$={self.hf.r_2:.3}, k ={self.hf.ec50:.3E}, n ={self.hf.nH:.3E}")
                            self.plots.titration_data["lims"].set_data([self.plots.min_pt, self.plots.max_pt], [
                                max_gain[concentration.index(self.plots.min_pt)],
                                max_gain[concentration.index(self.plots.max_pt)]])
                            self.plots.titration_data["lims"].set_label(f"Hill limits")

                        else:
                            self.linear_coefs = np.polyfit(concentration[
                                                           concentration.index(self.plots.min_pt):concentration.index(
                                                               self.plots.max_pt) + 1], max_gain[concentration.index(
                                self.plots.min_pt):concentration.index(self.plots.max_pt) + 1], 1)
                            fit_for_r2 = list(np.polyval(self.linear_coefs, concentration[concentration.index(
                                self.plots.min_pt):concentration.index(self.plots.max_pt) + 1]))
                            r_2 = r2_score(max_gain[concentration.index(self.plots.min_pt):concentration.index(
                                self.plots.max_pt) + 1], fit_for_r2)
                            self.plots.titration_data["titration"].set_data(concentration, max_gain)
                            self.plots.titration_data["fit"].set_data(concentration[concentration.index(
                                self.plots.min_pt):concentration.index(self.plots.max_pt) + 1], fit_for_r2)
                            self.plots.titration_data["fit"].set_label(
                                f"$R^2$={r_2:.3},a={self.linear_coefs[0]:.3}, b ={self.linear_coefs[1]:.3E}")
                            self.plots.titration_data["lims"].set_data([self.plots.min_pt, self.plots.max_pt], [
                                max_gain[concentration.index(self.plots.min_pt)],
                                max_gain[concentration.index(self.plots.max_pt)]])
                            self.plots.titration_data["lims"].set_label(f"Linear limits")

                        max_x = np.max(max_gain)
                        min_x = np.min(max_gain)
                        max_concentration = np.max(concentration)
                        min_concentration = np.min(concentration)

                        self.plots.titration.set_ylim(min_x - abs(min_x / 3), max_x + abs(min_x / 3))
                        self.plots.titration.set_xlim(min_concentration - abs(min_concentration / 3),
                                                      max_concentration + abs(min_concentration / 3))
                        self.plots.titration.legend().set_visible(True)
                else:
                    self.plots.reset_titration_graph()
                ##################################################### voltammogram Graph #######################################################
                if self.raw_data_df is not None and len(self.raw_data_df) != 0:
                    length = len(self.raw_data_df)
                    if self.chkBtn_show_latest_voltammogram_var.get():
                        self.datapoint_select_N = length - 1
                        self.volta_slider.set(self.datapoint_select_N + 1)
                        self.volta_slider.config(state="disabled", sliderlength=0)
                        self.volta_slider_text.config(text=f"{length}/{length}")
                    else:
                        self.volta_slider.config(state="active")
                        self.volta_slider.config(to=length, sliderlength=30, showvalue=False)
                        self.datapoint_select_N = self.volta_slider.get() - 1 if len(self.raw_data_df) > 0 else 0
                        self.volta_slider_text.config(text=f"{self.datapoint_select_N + 1}/{length}")

                    ################################################ change time format on axes ###################################
                    if self.update_raw_data_graph == True:
                        self.plots.rt_concentration.get_xaxis().set_major_formatter(
                            matplotlib.dates.DateFormatter('%y-%m-%d %H:%M:%S', tz=tz.gettz('America/Montreal')))
                        self.plots.rt_peak.get_xaxis().set_major_formatter(
                            matplotlib.dates.DateFormatter('', tz=tz.gettz('America/Montreal')))
                        _time = self.raw_data_df['time'].tolist()

                        ############################################### Voltammogram Graph ######################################
                        # try:
                        self.plots.volt_graph_data["raw_data"].set_data(
                            self.raw_data_df['raw_voltages'].iloc[self.datapoint_select_N][25:],
                            self.raw_data_df['raw_currents'].iloc[self.datapoint_select_N][25:])

                        try:
                            self.plots.volt_graph_data["smooth_data"].set_data(
                                self.raw_data_df['raw_voltages'].iloc[self.datapoint_select_N][25:],
                                self.raw_data_df['smooth_data'].iloc[self.datapoint_select_N][25:])
                            if len(self.plots.volt_graph_data["smooth_data"].get_xdata()) != len(self.plots.volt_graph_data["smooth_data"].get_ydata()):
                                self.plots.volt_graph_data["smooth_data"].set_data([],[])
                        except:
                            print("No smooth data")


                        baseline = []
                        normalized_gain = []
                        for i in range(len(self.raw_data_df['raw_voltages'].iloc[:])):
                            baseline.append(list(np.polyval(self.raw_data_df['baseline'].iloc[i],
                                                            self.raw_data_df['raw_voltages'].iloc[i])))
                            normalized_gain.append(list(np.polyval(self.raw_data_df['normalized_gain'].iloc[i],
                                                                   self.raw_data_df['raw_voltages'].iloc[i])))

                        self.plots.volt_graph_data["baseline"].set_data(
                            self.raw_data_df['raw_voltages'].iloc[self.datapoint_select_N][25:],
                            baseline[self.datapoint_select_N][25:])

                        # self.plots.gain_data["Gain"].set_data(self.raw_data_df['raw_voltages'].iloc[self.datapoint_select_N],normalized_gain[self.datapoint_select_N])

                        # Red line to show peak on voltammogram
                        index = normalized_gain[self.datapoint_select_N].index(max(normalized_gain[self.datapoint_select_N]))

                        self.plots.volt_graph.set_xlim(
                            self.raw_data_df['raw_voltages'].iloc[self.datapoint_select_N][25],
                            self.raw_data_df['raw_voltages'].iloc[self.datapoint_select_N][-1])


                        max_raw_current = [max(raw_curr[25:]) for raw_curr in self.raw_data_df['raw_currents']]
                        min_raw_current = [min(raw_curr[25:]) for raw_curr in self.raw_data_df['raw_currents']]
                        max_gain = [max(gain) for gain in normalized_gain]
                        min_gain = [min(gain) for gain in normalized_gain]
                        self.plots.gain_data['PeakX'].set_data(
                            [self.raw_data_df['peak_voltage'].iloc[self.datapoint_select_N],
                             self.raw_data_df['peak_voltage'].iloc[self.datapoint_select_N]],
                            [min(min_gain), max(max_gain)])

                        self.plots.volt_graph.set_ylim(min(min_raw_current), max(max_raw_current))
                        self.plots.gain.set_ylim(min(min_gain), max(max_gain))

                else:
                    self.plots.reset_rt_graphs()
                    self.volta_slider_text.config(text="No data in dataframe")
                    self.volta_slider.config(state="disabled", sliderlength=0)
                    self.plots.canvas.draw()
                    pass

                ###############################################  rt Graphs ####################################################
                if self.raw_data_df is not None and len(self.raw_data_df) != 0 and self.titration_df is not None:
                    if self.update_titration_graph == True or self.update_raw_data_graph == True:
                        h0 = []
                        h1 = []
                        _time = self.raw_data_df['time'].tolist()
                        for heights in list(self.raw_data_df['half_heigths'][:]):
                            h0.append(heights[0])
                            h1.append(heights[1])
                        _peak_voltage = list(self.raw_data_df['peak_voltage'][:])
                        _half_height0 = h0
                        _half_height1 = h1
                        self.plots.rt_peak_data["rt Peaks"].set_data(_time, _peak_voltage)
                        self.plots.rt_peak_data["rt Peaks max"].set_data(_time, _half_height0)
                        self.plots.rt_peak_data["rt Peaks min"].set_data(_time, _half_height1)
                        self.plots.rt_peak.set_xlim(min(_time), max(_time))
                        self.plots.rt_peak.set_ylim(self.raw_data_df['raw_voltages'].iloc[self.datapoint_select_N][0],
                                                    self.raw_data_df['raw_voltages'].iloc[self.datapoint_select_N][-1])

                        ########################################## rt Concentration ##########################################
                        if self.test_cBox.get() != 'CV':
                            try:
                                real_concentration = []
                                _t = []
                                first = True
                                for i in range(len(self.raw_data_df['raw_voltages'].iloc[:])):
                                    # normalized_gain = list(np.polyval(self.raw_data_df['normalized_gain'].iloc[i], self.raw_data_df['raw_voltages'].iloc[i]))
                                    # maximum_gain = np.max(normalized_gain)
                                    #
                                    baseline = list(np.polyval(self.raw_data_df['baseline'].iloc[i],
                                                               self.raw_data_df['raw_voltages'].iloc[i]))
                                    # g = self.raw_data_df['peak_current'].iloc[i] - baseline[
                                    #     list(self.raw_data_df['raw_currents'].iloc[i]).index(
                                    #         self.raw_data_df['peak_current'].iloc[i])]
                                    g = self.raw_data_df['peak_current'].iloc[i] #- baseline[list(self.raw_data_df['raw_currents'].iloc[i]).index(self.raw_data_df['peak_current'].iloc[i])]

                                    maximum_gain = g

                                    if first:
                                        first_peak_value = maximum_gain
                                        first = False
                                    # normalized
                                    maximum_gain = maximum_gain/first_peak_value
                                    maximum_gain = (maximum_gain - 1)*100

                                    if self.isHill:
                                        top, bottom, ec50, nH = self.hf.params
                                        if bottom <= maximum_gain <= top:
                                            if not np.isnan(ec50 * (((bottom - maximum_gain) / (maximum_gain - top)) ** (1 / nH))):
                                                real_concentration.append(ec50 * (((bottom - maximum_gain) / (maximum_gain - top)) ** (1 / nH)))
                                                _t.append(_time[i])
                                        print("Max_gain :", maximum_gain)
                                        print("Concentration :",ec50 * (((bottom - maximum_gain) / (maximum_gain - top)) ** (1 / nH)) )
                                    else:
                                        c = (maximum_gain - self.linear_coefs[1]) / self.linear_coefs[0]
                                        real_concentration.append(c)
                                        _t.append(_time[i])

                                    if self.test_cBox.get() == 'SWV':
                                        try:
                                            self.raw_data_df['concentration'].iloc[i] = real_concentration[-1]
                                        except IndexError:
                                            pass
                                if len(real_concentration) > 0:
                                    self.plots.rt_concentration.set_ylim(min(real_concentration),
                                                                         max(real_concentration))
                                    self.plots.rt_concentration.set_xlim(min(_t), max(_t))
                                    self.plots.rt_peak.set_xlim(min(_t), max(_t))
                                    self.plots.rt_concentration_data["rt concentration"].set_data(_t,real_concentration)

                            except Exception:
                                debug()
                                pass
                self.update_titration_graph = False
                self.update_raw_data_graph = False
                self.plots.fig.tight_layout()
                ############################################ Axes settings ##################################################

                if self.toggle_cursor:
                    self.cursors_v['peak_voltage_3'].set_data(
                        [self.raw_data_df['peak_voltage'].iloc[self.datapoint_select_N],
                         self.raw_data_df['peak_voltage'].iloc[self.datapoint_select_N]], [0, 1])

                    self.cursors_h['peak_current_3'].set_data(
                        [0, 1],
                        [self.raw_data_df['peak_current'].iloc[self.datapoint_select_N],
                         self.raw_data_df['peak_current'].iloc[self.datapoint_select_N]])
            except Exception as e:
                debug()
                pass
            else:
                try:
                    self.plots.canvas.draw()
                except Exception as e:
                    debug()
                    pass

    async def update_ui_loop(self, interval):
        """Updates UI, at regular intervals

        param interval: minimum time between 2 updates, time of execution is taken in account
        """
        self.print('UI started')
        waiter = StableWaiter(interval=interval)
        while True:
            try:
                await waiter.wait_async()
                self.update()
            except Exception as e:
                self.print(e)
                debug()
                messagebox.showerror('Error', e.__str__())

    async def start_scanning_process(self):
        """Starts scanning process"""
        try:
            self.stop_scanning_handle, self.dict_of_devices_global = await self.BLE_connector_instance.start_scanning()
            self.print('Scanning started')
            await asyncio.sleep(0.1)
        except Exception as e:
            self.print(e)
            debug()
            try:
                self.print('Stopping scanning because of an error')
                await self.stop_scanning_handle()
            except Exception as e2:
                self.print(e2)
                debug()
                messagebox.showerror('Error', e2.__str__())
            messagebox.showerror('Error', e.__str__())

    async def update_battery_loop(self, interval):
        """Updates battery voltage, at regular intervals

        param interval: minimum time between 2 updates, time of execution is taken in account
        """
        self.print('Battery update started')

        waiter = StableWaiter(interval=interval)
        while True:
            try:
                await waiter.wait_async()
                # self.update()
                voltage = await self.BLE_connector_instance.read_characteristic(
                    char_uuid='340a1b80-cf4b-11e1-ac36-0002a5d5c51b')
                if voltage is None:
                    continue
                # self.print(struct.unpack('<f', voltage))
            except Exception:
                pass
                # self.print(e)
                # debug()
                # messagebox.showerror('Error', e.__str__())

    async def autosave_loop(self, percentage_of_time):
        """Automatically saves data to file, at regular intervals

        param interval: minimum time between 2 updates, time of execution is taken in account
        """
        self.print('Auto save loop started')

        waiter = StableWaiter(percentage_of_time=percentage_of_time)
        i = 0
        while True:
            try:
                await waiter.wait_async_constant_avg_time()
                temp = [datetime.datetime.now().strftime('%x %X'), 'Autosaving...', i]
                self.print(temp)
                i += 1

                date = datetime.datetime.now().strftime('%Y-%m-%d')
                if date not in os.listdir(f"{self.output_path}"):
                    os.mkdir(f"{self.output_path}\\{date}")
                    os.makedirs(f"{self.output_path}\\{date}\\titrations")
                    os.makedirs(f"{self.output_path}\\{date}\\experiments")
                    os.makedirs(f"{self.output_path}\\{date}\\SW_experiments")

                save_temp = {}
                for df in self.curr_Titration_df.get_df_list():
                    save_temp[df] = self.curr_Titration_df.get_df_data(df).to_dict(orient='index')
                with open(f"{self.output_path}\\{date}\\titrations\\autosave.json",
                          'w') as f:  # overwrite previous autosave
                    json.dump(save_temp, f, indent=4, default=str)
                    f.close()

                save_temp = {}
                for df in self.curr_Experiment_df.get_df_list():
                    save_temp[df] = self.curr_Experiment_df.get_df_data(df).to_dict(orient='index')
                with open(f"{self.output_path}\\{date}\\experiments\\autosave.json",
                          'w') as f:  # overwrite previous autosave
                    json.dump(save_temp, f, indent=4, default=str)
                    f.close()

                save_temp = {}
                for df in self.SwiftMote_df.get_df_list():
                    save_temp[df] = self.SwiftMote_df.get_df_data(df).to_dict(orient='index')
                with open(f"{self.output_path}\\{date}\\SW_experiments\\autosave.json",
                          'w') as f:  # overwrite previous autosave
                    json.dump(save_temp, f, indent=4, default=str)
                    f.close()

            except Exception as e:
                self.print(e)
                debug()
                messagebox.showerror('Error', e.__str__())

    def print(self, txt):
        self.info_screen.config(state='normal')
        if type(txt) == list:
            temp = ""
            for i in txt:
                temp = temp + str(i) + " "
                self.info_screen.insert(tk.END, str(i) + " ")
            print(temp)
            self.info_screen.insert(tk.END, "\n")

        else:
            self.info_screen.insert(tk.END, str(txt) + "\n")
            print(str(txt))
        self.info_screen.yview_scroll(int(float(self.info_screen.index(tk.INSERT))), tk.UNITS)
        self.scroll_bar.set(float(self.info_screen.index(tk.INSERT)) - 20, float(self.info_screen.index(tk.INSERT)))
        self.info_screen.config(state='disabled')


class BLE_connector:
    def __init__(self, address="", to_connect=True):
        importlib.reload(bleak)  # to prevent deadlock
        self.address = address
        self.to_connect = to_connect
        try:
            asyncio.get_running_loop().run_until_complete(self.client.disconnect())
        except Exception as e:
            pass
        try:
            del self.client
        except Exception as e:
            pass
        if self.to_connect:
            self.client = bleak.BleakClient(address)
            self.connected_flag = False
            # asyncio.get_running_loop().run_until_complete(self.client.pair(1))

    async def keep_connections_to_device(self, uuids, callbacks):
        assert len(uuids) == len(callbacks)  # length and respective order must be the same,
        # the same function may be used twice with different UUIDs
        # (eg if there are 2 similar electrodes generating similar data at the same time)
        while True:
            try:
                if self.to_connect:
                    # workaround, without this line it sometimes cannot reconnect or takes a lot of time to reconnect
                    self.__init__(self.address, self.to_connect)
                    await self.client.connect(timeout=32)  # timeout should be the same as in firmware
                    if self.client.is_connected:
                        print("Connected to Device")
                        self.connected_flag = True

                        def on_disconnect(client):
                            print("Client with address {} got disconnected!".format(client.address))
                            self.connected_flag = False

                        self.client.set_disconnected_callback(on_disconnect)
                        for uuid, callback in zip(uuids, callbacks):
                            await self.client.start_notify(uuid, callback)
                        while True:
                            if not self.client.is_connected or not self.connected_flag:
                                print("Lost connection, reconnecting...")
                                await self.client.disconnect()
                                break
                            # else:
                            #     await self.test()

                            await asyncio.sleep(1)
                    else:
                        print(f"Not connected to Device, reconnecting...")

            except Exception as e:
                print(e)
                debug()
                print("Connection error, reconnecting...")
                await self.client.disconnect()  # accelerates reconnection
            self.connected_flag = False
            await asyncio.sleep(1)

    # async def scan(self):
    #    try:
    #        devices_list = []
    #
    #        devices = await bleak.BleakScanner.discover(5)
    #        devices.sort(key=lambda x: -x.rssi)  # sort by signal strength
    #        for device in devices:
    #            devices_list.append(str(device.address) + "/" + str(device.name) + "/" + str(device.rssi))
    #        #
    #        return devices_list
    #
    #        # scanner = bleak.BleakScanner()
    #        # scanner.register_detection_callback(self.detection_callback)
    #        # await scanner.start()
    #        # await asyncio.sleep(5.0)
    #        # await scanner.stop()
    #
    #
    #    except Exception as e:
    #        print(e)

    # def detection_callback(device, advertisement_data):
    #    print(device.address, "RSSI:", device.rssi, advertisement_data)
    async def start_scanning(self):
        try:
            dict_of_devices = {}

            def detection_callback(device, advertisement_data):
                # print(device.address, "RSSI:", device.rssi, advertisement_data)
                dict_of_devices[device.address] = device  # overwrites device object

            scanner = bleak.BleakScanner(scanning_mode="passive")
            scanner.register_detection_callback(detection_callback)
            await scanner.start()

            return scanner.stop, dict_of_devices

        except Exception as e:
            print(e)
            debug()
            return -1, -1

    async def read_characteristic(self, char_uuid='340a1b80-cf4b-11e1-ac36-0002a5d5c51b'):
        try:
            if self.connected_flag:
                return await self.client.read_gatt_char(char_uuid)
            return None
        except Exception as e:
            print(e)
            debug()
            return None

    async def write_characteristic(self, char_uuid="330a1b80-cf4b-11e1-ac36-0002a5d5c51b", data=b"Hello World!"):
        try:
            if self.connected_flag:
                return await self.client.write_gatt_char(char_uuid,
                                                         data,
                                                         response=True
                                                         )
            return None
        except Exception as e:
            print(e)
            debug()
            return None

    async def read_all_characteristics(self):
        services = await self.client.get_services()
        for characteristic in services.characteristics.values():
            try:
                print(characteristic.uuid, await self.client.read_gatt_char(characteristic))
            except Exception as e:
                pass

    async def test(self):
        print("test")
        print(self.client.mtu_size)
        try:
            print("qwer")
            print(await self.client.write_gatt_char("330a1b80-cf4b-11e1-ac36-0002a5d5c51b",
                                                    b"Hello World!",
                                                    response=True
                                                    )
                  )
            print("qwer2")
            await asyncio.sleep(0.1)
            # a = await self.client.get_services()  #
            # # print(a)
            # # b=a.descriptors.values()
            # # print(b)
            # for i, c in enumerate(a.characteristics.values()):
            #     # print(c.uuid, c.__dict__)
            #     #
            #     print(i, c.uuid, c.properties, c.__dict__)
            #     if "write" not in c.properties:
            #         continue
            #     try:
            #         # self.client.p
            #         # await self.client.pair(1)
            #         # await self.client.write_gatt_descriptor(c, B"123ABC")
            #         print("qwer")
            #         print(await self.client.write_gatt_char(c, bytearray(b'\x02\x03\x05\x07'), response=True))
            #         await asyncio.sleep(0.1)
            #         print(await self.client.read_gatt_char(c))
            #         await asyncio.sleep(0.1)
            #         # bytearray(b'\x02\x03\x05\x07')
            #         # print(b)
            #     #
            #     except Exception as e:
            #         print("Test error 2:", e)
            #         if "Access Denied" not in str(e):
            #             print("Have a look!", e)
            #     await asyncio.sleep(0.1)
            # # '330a1b80-cf4b-11e1-ac36-0002a5d5c51b'
            # # print(a.characteristics[20])
        except Exception as e:
            debug()
            print("Test error:", e)

    async def disconnect(self):
        try:
            if self.client.is_connected:
                print("Disconnecting...")
                # del self.client
                await self.client.disconnect()
                print("Disconnected")
        except Exception as e:
            # debug()
            pass


class StableWaiter:
    """Generates intervals between executions of certain parts of code;
    two methods: constant time, and percentage of total execution time """

    def __init__(self, interval=1.0, percentage_of_time=10):
        self.interval = interval
        self.duty_cycle = percentage_of_time / 100
        self.t1 = datetime.datetime.now(datetime.timezone.utc)

    async def wait_async(self):
        """Waits at approximately the same intervals independently of CPU speed
        (if CPU is faster than certain threshold)
        This is not mandatory, but makes UI smoother
        Can be roughly simplified with asyncio.sleep(interval)"""

        t2 = datetime.datetime.now(datetime.timezone.utc)
        previous_frame_time = ((t2 - self.t1).total_seconds())
        self.t1 = t2
        await asyncio.sleep(min((self.interval * 2) - previous_frame_time, self.interval))

    async def wait_async_constant_avg_time(self):
        """Waits constant average time as a percentage of total execution time
        O(1) avg difficulty, used to accelerate O(N^2) or worse algorithms by running them less frequently as N increases
        This is not mandatory, but makes UI smoother
        Can be roughly simplified with asyncio.sleep(interval), for example, it is used by autosaving in this app"""

        t2 = datetime.datetime.now(datetime.timezone.utc)
        previous_frame_time = ((t2 - self.t1).total_seconds())
        self.t1 = t2

        await asyncio.sleep(previous_frame_time / self.duty_cycle - previous_frame_time)


class Packet:
    transaction_number_bytes = 1  # use 1 byte to represent transaction field
    packet_number_bytes = 1  # use 1 byte to represent packet field
    time_number_bytes = 4  # 4*1-byte fields represent time: hours, minutes, seconds, microseconds
    metadata_length_total_bytes = transaction_number_bytes + packet_number_bytes + time_number_bytes
    datapoint_length_bytes = 2  # each data point is 2 bytes

    def __init__(self, data: bytearray, time_delivered):
        """Parse packet"""
        self.data = data
        self.time_delivered = time_delivered
        # self.datahex=data.hex()

        # self.print(data.hex())

        self.transaction_number = struct.unpack('<B', self.data[0:0 + self.transaction_number_bytes])[0]
        self.packet_number = struct.unpack('<B', self.data[1:1 + self.packet_number_bytes])[0]
        time_packet_created = struct.unpack('<BBBB', self.data[2:2 + self.time_number_bytes])[::-1]
        # time_packet_created.reverse()
        # self.print(time_transmitted)

        # transmit only 24 hours of time, year/month/date is not transmitted since experiment lasts only 6 hours,
        # modify if longer interval is needed with no added jitter,
        # but it is not required since overflow will lead to auto adjustment of offset
        self.time_created = datetime.datetime(year=2000, month=1, day=1,
                                              hour=time_packet_created[0],
                                              minute=time_packet_created[1],
                                              second=time_packet_created[2],
                                              microsecond=round(
                                                  1000000 * (math.pow(2, 8) - time_packet_created[3] - 1) /
                                                  (math.pow(2, 8) + 1)
                                              ),
                                              tzinfo=datetime.timezone.utc
                                              )  # .timestamp()
        # cself.print(self.time_transmitted_datetime)

        length = len(data) - self.metadata_length_total_bytes  # 2 bytes are metadata
        number_of_datapoints = math.floor(length / self.datapoint_length_bytes)  # 2 bytes per datapoint

        self.datapoints = [-1] * number_of_datapoints  # initialize list of datapoints

        for i in range(number_of_datapoints):
            self.datapoints[i] = struct.unpack('<H',
                                               self.data[
                                               self.metadata_length_total_bytes + self.datapoint_length_bytes * i:
                                               self.metadata_length_total_bytes + self.datapoint_length_bytes * (i + 1)
                                               ])[0]

    def get_datapoints(self):
        """Data load of the BLE packet"""
        return self.datapoints


class Transaction:
    """One indivisible piece of useful data
    2 modes of operation:
    1) Size is known
    2) Size is unknown
    """

    def __init__(self, size=0):
        self.size = size
        self.packets: {Packet} = {}
        self.transaction_number = -1
        self.finalized = False

    def add_packet(self, data: bytearray, time_delivered):
        if self.finalized:
            # self.print("Error, this transaction is already finalized")
            return -1

        packet = Packet(data=data, time_delivered=time_delivered)  # create a Packet object

        if self.transaction_number == -1:
            # self.print("First packet of new transaction received")
            self.transaction_number = packet.transaction_number

        if self.transaction_number == packet.transaction_number:
            # self.print("Adding new packet")
            if packet.packet_number not in self.packets:
                self.packets[packet.packet_number] = packet
            else:
                self.print("Error, this packet was already received")
                return -1
        else:
            if self.size != 0:  # if size is not set, estimate number of packets.
                self.print("Transaction probably finished successfully")
                self.finalized = True
                self.size = len(self.packets)
                self.print("Transaction size")
                return -2
            else:
                self.print("Error, Transaction number is different, this should never happen")
                return -1

        if len(self.packets) == self.size:
            self.print("Transaction finished successfully")
            self.finalized = True
            return 0
        else:
            return 1  # continue waiting for more packets

    def get_joined_data(self):
        try:
            if self.finalized:
                all_datapoints = []
                for i in range(self.size):
                    all_datapoints.extend(self.packets[i].get_datapoints())

                # removes 0s at the end, hopefully it does not delete useful data
                while len(all_datapoints) >= 1 and all_datapoints[-1] == 0:
                    all_datapoints.pop(-1)

                self.print(all_datapoints)

                return all_datapoints
            else:
                # self.print("Error, not finalized yet")
                return None
        except Exception:
            return None

    def get_times_of_delivery(self):
        # should be in ascending order, but no checks are done
        if self.finalized:
            all_times_of_delivery = {}
            for i in range(self.size):
                all_times_of_delivery[i] = self.packets[i].time_delivered
            return all_times_of_delivery
        else:
            # self.print("Error, not finalized yet")
            return None

    def get_min_time_of_transaction_delivery(self):
        if self.finalized:
            return min(self.get_times_of_delivery().values())
        else:
            return None

    def get_times_of_packet_creation(self):  # for debugging
        # should be in ascending order, but no checks are done
        if self.finalized:
            all_times_of_transmitting = {}
            for i in range(self.size):
                all_times_of_transmitting[i] = self.packets[i].time_created
            return all_times_of_transmitting
        else:
            # self.print("Error, not finalized yet")
            return None

    def get_min_time_of_transaction_creation(self):
        if self.finalized:
            return min(self.get_times_of_packet_creation().values())
        else:
            return None

    def print(self, all_datapoints):
        pass


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    app = App(loop)
    loop.run_forever()