import matplotlib
import matplotlib.pyplot as plt
from Utils import debug
from dateutil import tz
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

class Plot():
    prev_min_pt = None
    prev_max_pt = None

    def __init__(self):
        """Initializes plots
                param self: reference to parent object
                """
        self.min_pt = 10000
        self.max_pt = -10000

        plt.rcParams['axes.grid'] = True  # enables all grid lines globally

        self.fig = plt.figure(dpi=100)
        ######################################################## voltammogram graph ###############################################################
        self.volt_graph = self.fig.add_subplot(2, 2, 1)
        self.volt_graph_data = {"raw_data": self.volt_graph.plot([], [], '-', color='g')[0],
                                "smooth_data": self.volt_graph.plot([], [], '-', color='r')[0],
                                "baseline": self.volt_graph.plot([], [], '-', color='c')[0]}
        
        self.gain = self.volt_graph.twinx()
        #self.gain.set_ylabel("Normalized gain (%)", color='b')
        self.gain_data = {
                      "Gain": self.gain.plot([], [], '-', color='b')[0],
                      "PeakX": self.gain.plot([], [], '-', color='r')[0],
                      "PeakY": self.gain.plot([], [], '-', color='r')[0],
                      }
        self.volt_graph.set_xlabel("Stimulus (mV)")
        self.volt_graph.set_ylabel("Current (µA)", color='g')  # Filtered current is on the same plot
        self.volt_graph.get_xaxis().set_major_locator(matplotlib.ticker.AutoLocator())
        self.volt_graph.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        self.volt_graph.get_yaxis().set_major_locator(matplotlib.ticker.AutoLocator())
        self.volt_graph.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ######################################################## real time peaks graph ###############################################################
        self.rt_peak = self.fig.add_subplot(2, 2, 2)
        self.rt_peak_data = {   "rt Peaks": self.rt_peak.plot([], [], '-', color="b")[0],
                                "rt Peaks max": self.rt_peak.plot([], [], '-', color="c")[0],
                                "rt Peaks min": self.rt_peak.plot([], [], '-', color="c")[0],
                                "current rt Peak": self.rt_peak.plot([], [], '-', color="r")[0]
                                }
        self.rt_peak.set_xlabel("Time (h)")
        self.rt_peak.set_ylabel("Peak Voltage(mV)")  # Battery/signal strength can be also on this plot
        self.rt_peak.get_xaxis().set_major_formatter(matplotlib.dates.DateFormatter('', tz=tz.gettz('America/Montreal')))
        self.rt_peak.get_xaxis().set_major_locator(matplotlib.dates.AutoDateLocator())
        self.rt_peak.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        self.rt_peak.get_yaxis().set_major_locator(matplotlib.ticker.AutoLocator())
        self.rt_peak.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        plt.setp(self.rt_peak.get_xticklabels(), rotation=45, horizontalalignment='right')
        ####################################################### titration graph ###############################################################
        self.titration = self.fig.add_subplot(2, 2, 3)
        self.titration.get_yaxis().set_major_locator(matplotlib.ticker.AutoLocator())
        self.titration.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        self.titration.get_xaxis().set_major_locator(matplotlib.ticker.AutoLocator())
        self.titration.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        self.titration_data ={
                                "titration": self.titration.semilogx([], [], '-', color='b', label="titration curve")[0],
                                "fit": self.titration.semilogx([], [], '-', color='r', label="Hill fit")[0],
                                "lims": self.titration.semilogx([], [], 'ro', color='r', label="Hill limits")[0]
                                }
        self.titration.set_xlabel("Concentration (mol/L)")
        self.titration.set_ylabel("Gain (%)")

        ####################################################### real time concentration graph ###############################################################
        self.rt_concentration = self.fig.add_subplot(2, 2, 4)
        self.rt_concentration_data = {"rt concentration": self.rt_concentration.semilogy([], [], '-')[0]}

        self.rt_concentration.get_xaxis().set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M:%S', tz=tz.gettz('America/Montreal')))
        self.rt_concentration.get_xaxis().set_major_locator(matplotlib.dates.AutoDateLocator())
        self.rt_concentration.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        plt.setp(self.rt_concentration.get_xticklabels(), rotation=45, horizontalalignment='right')
        self.rt_concentration.set_xlabel("Time")
        self.rt_concentration.set_ylabel("Concentration (mol/L)")
        #####################################################################################################################################################

        # frame_plots = tk.Frame(master=frame)
        # frame_controls = tk.Frame(master=frame)
        # frame_slider = tk.Frame(master=frame_controls)
        # frame_toolbar = tk.Frame(master=frame_controls)
        #
        # self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plots)  # A tk.DrawingArea.
        # self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # self.toolbar = NavigationToolbar2Tk(canvas=self.canvas, window=frame_toolbar, pack_toolbar=False)

        # # remove some buttons at the bottom of the plot part of the window
        # self.toolbar.toolitems = (('Pan', 'Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect', 'move', 'pan'),
        #                           ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
        #                           ('Save', 'Save the figure', 'filesave', 'save_figure'))

        # # Reinitialize the toolbar to apply button changes. This is a bit of a hack. Change if there is a better way.
        # self.toolbar.__init__(canvas=self.canvas,
        #                       window=frame_toolbar,
        #                       pack_toolbar=False
        #                       )
        # self.toolbar.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)

        # def onclick(event):
        #     try:
        #         if event.inaxes is not None:
        #             if event.inaxes is self.titration:
        #                 if master.titration_df is not None:
        #                     x = event.xdata
        #                     concentrations = (self.titration_data["titration"].get_data())[0]
        #                     mid_val = np.median(concentrations)
        #                     minimum = {_conc: abs(_conc - x) for _conc in concentrations}
        #                     values = list(minimum.values())
        #                     min_val = np.min(list(minimum.values()))
        #                     conc_ = list(minimum.keys())[values.index(min_val)]
        #                     if x <= mid_val:
        #                         self.min_pt = conc_
        #                     else:
        #                         self.max_pt = conc_
        #
        #                     print("Updating range")
        #                     Plot.prev_min_pt = self.min_pt
        #                     Plot.prev_max_pt = self.max_pt
        #
        #                 master.update_titration_graph = True
        #                 master.to_update_plots = True
        #     except Exception as e:
        #         debug()
        #
        # self.canvas.mpl_connect('button_press_event', onclick)
        #
        # def apply_tight_layout(event: tk.Event):
        #     """when resize the whole window, the plot part of the window is resized """
        #     try:
        #         if str(event.widget) == ".!frame.!frame.!canvas":
        #             self.fig.tight_layout()
        #     except Exception as e:
        #         master.print(e)
        #         pass
        #
        # master.bind("<Configure>", apply_tight_layout)  # resize plots when window size changes, "Configure" comes from tk
        #
        # frame_plots.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor='center')
        # frame_controls.pack(side=tk.TOP, fill=tk.BOTH, expand=False, anchor='center')
        # frame_controls.grid_columnconfigure(0, weight=1, uniform="group1")
        # frame_controls.grid_columnconfigure(1, weight=1, uniform="group1")
        # frame_controls.grid_rowconfigure(0, weight=1)
        # frame_toolbar.grid(row=0, column=0, sticky="nsew", rowspan=2)
        # frame_slider.grid(row=0, column=1, sticky="ns")

    def reset_titration_graph(self):
        self.titration_data["titration"].set_data([], [])
        self.titration_data["fit"].set_data([], [])
        self.titration_data["lims"].set_data([], [])
        self.titration.legend().set_visible(False)

    def reset_rt_graphs(self):
        self.volt_graph_data["raw_data"].set_data([], [])
        self.volt_graph_data["smooth_data"].set_data([], [])
        self.volt_graph_data["baseline"].set_data([], [])
        self.gain_data["Gain"].set_data([], [])
        self.gain_data["PeakX"].set_data([], [])
        self.gain_data["PeakY"].set_data([], [])
        self.rt_peak_data["rt Peaks"].set_data([], [])
        self.rt_peak_data["rt Peaks max"].set_data([], [])
        self.rt_peak_data["rt Peaks min"].set_data([], [])
        self.rt_peak_data["current rt Peak"].set_data([], [])
        self.rt_concentration_data["rt concentration"].set_data([], [])
