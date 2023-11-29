import asyncio
import importlib
import json
import math
import struct
import warnings
import time
import threading
import re

import bleak
import matplotlib
import nest_asyncio
import BLE_connector_Bleak
from Process_CH_data import *
from Data_processing import *
import pickle
import datetime
from Tests import Test
from Plots import Plot
from values import Experiment_c
from values import Electrode_c
from values import Titration_c
from values import Test_c

# hotfix to run nested asyncio to correctly close Bleak without having to wait for timeout to reconnect to device again
nest_asyncio.apply()

address_default = 'FE:B7:22:CC:BA:8D'
uuids_default = ['340a1b80-cf4b-11e1-ac36-0002a5d5c51b', ]
write_uuid = '330a1b80-cf4b-11e1-ac36-0002a5d5c51b'

sender_battery_voltage = "battery_voltage"
SUPPRESS_WARNINGS = True
if SUPPRESS_WARNINGS:
    warnings.filterwarnings("ignore")


class App():

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
        self.first_gain = 0
        self.first_measure = True

        ################################################# Menu bar #######################################################
        self.tasks = {}  # list of tasks to be continuously executed at the same time (asynchronously, not in parallel)

        ###############################################################################################################
        self.plots = Plot()
        self.init_controls()

        #################################################################
        # Testing purposes
        self.time_type = True
        #################################################################

    def create_directories(self, dir_list):
        for path in dir_list:
            if not os.path.exists(path):
                os.makedirs(path)

    def init_controls(self):

        """Initializes controls param master: reference to parent object """
        self.current_values = {}  # values from variable fields that can be sent to the device
        self.option_cbox_values = {}
        self.graph_values = {}
        self.calib_data_dict = {}


        ############################ Electrode selection with dropdown Experiment updating #############################################
        def update_electrode_list():
            elec_list = list(os.listdir(f"{self.data_path}"))
            self.Electrode_cBox["values"] = elec_list

        self.Electrode_cBox = Electrode_c()

        def load_electrode(name):
            with open(f"{self.data_path}\\{name}", "rb") as f:
                self.electrode_list[name] = pickle.load(f)

        def load_titration(name):
            self.update_titration_graph = True
            with open(f"{self.titration_path}\\{name}", "rb") as f:
                self.titration_list[name] = pickle.load(f)

        def set_new_electrode():
            electrode_name = self.Electrode_cBox.get()
            load_electrode(electrode_name)
            self.current_electrode = self.electrode_list[electrode_name]
            self.Experiment_cBox.set("")
            self.titration_df = None
            self.test_cBox.set("")
            self.raw_data_df = None
            self.to_update_plots = True

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
            load_electrode(electrode_name)
            self.current_electrode = self.electrode_list[electrode_name]
            self.Experiment_cBox.set("")
            self.titration_df = None
            self.test_cBox.set("")
            self.raw_data_df = None
            self.to_update_plots = True

        # ############################# Experiment selection from selected Electrode ######################################################
        self.Experiment_cBox = Experiment_c()

        # ############################# Titration selection from file ######################################################
        def update_titration_list():
            titr_list = list(os.listdir(f"{self.titration_path}"))
            self.Titration_cBox['values'] = titr_list

        def force_set_titration():
            self.Titration_cBox.set("take6")
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
                self.to_update_plots = True
            pass

        self.Titration_cBox = Titration_c()

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
                    print(f"{name} created successfully")
                    electrode.save(self.data_path)

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
                print(f"{experiment_name} doesn't exist")
            else:
                if self.current_electrode.get_tests(experiment_name)["Titration"].get_df().shape[0] > 0:
                    self.titration_df = self.current_electrode.get_tests(experiment_name)["Titration"].get_df().sort_values(by=["concentration"])
                    self.plots.min_pt = list(self.titration_df["concentration"])[0]
                    self.plots.max_pt = list(self.titration_df["concentration"])[-1]
                    self.update_titration_graph = True
            self.to_update_plots = True

        # ############################# Experiment type selection from selected Experiment ######################################################
        def update_test_list():
            self.test_cBox["values"] = list(self.current_electrode.get_tests(self.Experiment_cBox.get()).keys())

        self.test_cBox = Test_c()

        def set_test_graph():
            self.raw_data_df = self.current_electrode.get_tests(self.Experiment_cBox.get())[self.test_cBox.get()].get_df()
            self.plots.rt_concentration_data["rt concentration"].set_data([], [])
            self.update_raw_data_graph = True
            self.to_update_plots = True

        ############################################### Results ##################################################
        def update_plot():
            self.to_update_plots = True

        def refresh_slider(event):
            if self.test_cBox.get() != "":
                update_raw_data_graph(event)

        def update_raw_data_graph(event):
            self.update_raw_data_graph = True
            self.to_update_plots = True

        def Force_Create_SWV():
            return self.current_electrode.get_tests(self.Experiment_cBox.get())["SWV"]

        def run_test_thread(test, comport):
            self.thread_result = test.run_test(comport, 115200)

        def handle_test_results_delayed(test):
            while self.thread_result == -1:
                time.sleep(0.1)  # Sleep for 100ms
            handle_test_results(test)  # Call handle_test_results once result is not -1
            self.data_received = True

        def handle_test_results(test):
            if self.thread_result == 1:
                self.current_electrode.save(self.data_path)
                print("Test ran successfully")
                set_test_graph()
                self.calculate_concentration()
                update_plot()
            elif not self.continuous_running:
                messagebox.showerror('Error 2', self.thread_result.__str__())
            else:
                if self.thread_result == "Test stopped by user":
                    messagebox.showerror('Error', self.thread_result.__str__())
                else:
                    print('Error', self.thread_result.__str__())
            self.thread_result = -1

        def run_test(test: Test):
            print("TEST ", test)
            self.test_runned = True
            self.thread_result == -1
            self.data_received = False
            ## ajout comm8
            comport = "COM6"
            threading.Thread(target=run_test_thread, args=(test, comport)).start()
            handle_test_results_delayed(test)

        def run_continuous_test(test:Test):
            test.stop_continuous = False
            self.continuous_running = True

            def run_test_and_update_gui():
                try:
                    run_test(test)
                    index = 0
                    if index < test.get_params()["RunTime"] and not test.stop_continuous:
                        while self.data_received == False:
                            pass
                        run_test_and_update_gui()
                    else:
                        self.continuous_running = False
                except Exception as e:
                    messagebox.showerror('Error 2', e.__str__())

            run_test_and_update_gui()  # Start the continuous test

        ###Ajout
        start_electrode()
        ######################################## Info screen params ########################################################################

        ################################################################################################################
        ###AJOUT
        start_experiment()
        force_set_titration()
        self.isHill = False
        self.prepare_titration()
        test = Force_Create_SWV()
        self.test_cBox.set("SWV")
        time.sleep(1)
        run_continuous_test(test)

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

    def prepare_titration(self):
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
                max_gain = [(x - 1) * 100 for x in max_gain]
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

    def calculate_concentration(self):
        try:
            i = len(self.raw_data_df['raw_voltages']) - 1  # Index for the last row
            g = self.raw_data_df['peak_current'].iloc[i]
            maximum_gain = g
            if self.first_measure:
                self.first_gain = maximum_gain
                self.first_measure = False
            # normalized
            maximum_gain = maximum_gain / self.first_gain
            maximum_gain = (maximum_gain - 1) * 100

            concentration = "NA"
            if self.isHill:
                top, bottom, ec50, nH = self.hf.params
                print("top ", top, "bottom ", bottom)
                print("max gain ", maximum_gain)
                if bottom <= maximum_gain <= top:
                    if not np.isnan(ec50 * (((bottom - maximum_gain) / (maximum_gain - top)) ** (1 / nH))):
                        concentration = ec50 * (((bottom - maximum_gain) / (maximum_gain - top)) ** (1 / nH))
                else:
                    concentration = "outRange"
            else:
                concentration = (maximum_gain - self.linear_coefs[1]) / self.linear_coefs[0]

            print("Concentration: ", concentration)
        except Exception:
            debug()
            pass

    async def start_scanning_process(self):
        """Starts scanning process"""
        try:
            self.stop_scanning_handle, self.dict_of_devices_global = await self.BLE_connector_instance.start_scanning()
            print('Scanning started')
            await asyncio.sleep(0.1)
        except Exception as e:
            print(e)
            debug()
            try:
                print('Stopping scanning because of an error')
                await self.stop_scanning_handle()
            except Exception as e2:
                print(e2)
                debug()
                messagebox.showerror('Error', e2.__str__())
            messagebox.showerror('Error', e.__str__())

    async def autosave_loop(self, percentage_of_time):
        """Automatically saves data to file, at regular intervals

        param interval: minimum time between 2 updates, time of execution is taken in account
        """
        print('Auto save loop started')

        waiter = StableWaiter(percentage_of_time=percentage_of_time)
        i = 0
        while True:
            try:
                await waiter.wait_async_constant_avg_time()
                temp = [datetime.datetime.now().strftime('%x %X'), 'Autosaving...', i]
                print(temp)
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
                print(e)
                debug()
                messagebox.showerror('Error', e.__str__())


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
                print("Error, this packet was already received")
                return -1
        else:
            if self.size != 0:  # if size is not set, estimate number of packets.
                print("Transaction probably finished successfully")
                self.finalized = True
                self.size = len(self.packets)
                print("Transaction size")
                return -2
            else:
                print("Error, Transaction number is different, this should never happen")
                return -1

        if len(self.packets) == self.size:
            print("Transaction finished successfully")
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

                print(all_datapoints)

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