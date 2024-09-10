"""
Live phase retrieval GUI.
"""

from pathlib import Path
import sys
sys.path.append(f"{Path(__file__).parents[1]}")

import time
import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter.filedialog import askdirectory
import tkinter.ttk as ttk

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

import src.phasing as phasing
import src.diffraction as diffraction
import src.utils as ut


DATA = 0
MANUAL = 1
AUTO = 2

UNITS = {power: unit for power, unit in zip([-4, -3, -2, -1, 0], ["pm", "nm", "μm", "mm", "m"])}

class App:
    def __init__(self):
        self.data = diffraction.LoadData()
        self.solver = phasing.Solver(self.data.preprocess())

        self.root = tk.Tk()
        self.root.title("Interactive Phase Retrieval (v 0.6)")

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        self.control_panel = ttk.Notebook(self.root)
        self.control_panel.grid(row=1, column=1, sticky=tk.NSEW)

        btn_kwargs = {"sticky": tk.EW, "padx": 2, "pady": 2}
        sep_kwargs = {"column": 0, "columnspan": 3, "sticky": tk.EW, "pady": 3}

        # Data controls ###############################################################################################
        data_tab = ttk.Frame(self.control_panel)

        r = 0
        ttk.Button(data_tab, text="Load data", command=self.load_data).grid(row=r, column=0, columnspan=3,
                                                                              **btn_kwargs)
        r += 1
        ttk.Button(data_tab, text="Load background", command=self.load_bkgd).grid(row=r, column=0, columnspan=3,
                                                                                  **btn_kwargs)
        r += 1
        self.det_pitch = tk.DoubleVar(value=5.5)
        self.det_dist = tk.DoubleVar(value=100)
        self.wavelength = tk.DoubleVar(value=532)
        params = ttk.LabelFrame(data_tab, text="Detector info", borderwidth=5)
        params.grid(row=r, column=0, columnspan=3, sticky=tk.S)
        self.det_params_entries = []  # widgets that are unlocked with the Edit button
        for row, label, var, unit in zip(
            [0, 1, 2],
            ["Pixel pitch : ", "Distance : ", "Wavelength : "],
            [self.det_pitch, self.det_dist, self.wavelength],
            ["μm", "mm", "nm"]
        ):
            ttk.Label(params, text=label, justify=tk.RIGHT).grid(row=row, column=0, sticky=tk.E)
            self.det_params_entries.append(ttk.Entry(params, textvariable=var, width=6))
            self.det_params_entries[-1].grid(row=row, column=1, padx=1)
            ttk.Label(params, text=unit, justify=tk.CENTER).grid(row=row, column=2)
        self.det_params_edit_button = ttk.Button(params, text="Edit", command=self.edit_det_params)
        self.det_params_edit_button.grid(row=3, column=0, **btn_kwargs)
        self.det_params_lock_button = ttk.Button(params, text="Set", command=self.lock_det_params)
        self.det_params_lock_button.grid(row=3, column=1, columnspan=2, **btn_kwargs)
        for widget in [*self.det_params_entries, self.det_params_lock_button]:
            widget["state"] = "disabled"

        r += 1
        ttk.Separator(data_tab, orient="horizontal").grid(row=r, **sep_kwargs)
        r += 1
        ttk.Label(data_tab, text="Image processing", font=('Arial', 12, 'underline')).grid(row=r, **sep_kwargs)
        r += 1
        self.pre_bkgd = tk.BooleanVar(value=False)
        ttk.Checkbutton(data_tab, text="Subtract background", variable=self.pre_bkgd).grid(row=r, column=0,
                                                                                           columnspan=3, **btn_kwargs)
        r += 1
        self.pre_bin_q = tk.BooleanVar(value=False)
        self.pre_bin_factor = tk.IntVar(value=1)
        ttk.Checkbutton(data_tab, text="Bin pixels", variable=self.pre_bin_q).grid(row=r, column=0, **btn_kwargs)
        FormatLabel(data_tab, textvariable=self.pre_bin_factor, format="{:.0f}").grid(row=r, column=1, **btn_kwargs)
        r += 1
        ttk.Scale(data_tab, from_=1, to=8, orient="horizontal", variable=self.pre_bin_factor).grid(
            row=r, column=0, columnspan=2, **btn_kwargs)

        r += 1
        self.pre_crop_q = tk.BooleanVar(value=False)
        self.pre_crop_factor = tk.DoubleVar(value=1)
        ttk.Checkbutton(data_tab, text="Crop pixels", variable=self.pre_crop_q).grid(row=r, column=0, **btn_kwargs)
        FormatLabel(data_tab, textvariable=self.pre_crop_factor, format="{:.2f}").grid(row=r, column=1, **btn_kwargs)
        r += 1
        ttk.Scale(data_tab, from_=1, to=0.1, orient="horizontal", variable=self.pre_crop_factor).grid(
            row=r, column=0, columnspan=2, **btn_kwargs)

        r += 1
        self.pre_gauss_q = tk.BooleanVar(value=False)
        self.pre_gauss_sigma = tk.DoubleVar(value=0.0)
        ttk.Checkbutton(data_tab, text="Gaussian blur", variable=self.pre_gauss_q).grid(row=r, column=0, **btn_kwargs)
        FormatLabel(data_tab, textvariable=self.pre_gauss_sigma, format="{:.2f}").grid(row=r, column=1, **btn_kwargs)
        r += 1
        ttk.Scale(data_tab, from_=0.0, to=5.0, orient="horizontal", variable=self.pre_gauss_sigma).grid(
            row=r, column=0, columnspan=2, **btn_kwargs)

        r += 1
        self.pre_threshold_q = tk.BooleanVar(value=False)
        self.pre_threshold_val = tk.DoubleVar(value=0.0)
        ttk.Checkbutton(data_tab, text="Threshold", variable=self.pre_threshold_q).grid(row=r, column=0,
                                                                                        **btn_kwargs)
        FormatLabel(data_tab, textvariable=self.pre_threshold_val, format="{:.2f}").grid(row=r, column=1, **btn_kwargs)
        r += 1
        ttk.Scale(data_tab, from_=0.0, to=1.0, orient="horizontal", variable=self.pre_threshold_val).grid(
            row=r, column=0, columnspan=2, **btn_kwargs)

        for var in [self.pre_bkgd, self.pre_bin_q, self.pre_bin_factor, self.pre_crop_q, self.pre_crop_factor,
                    self.pre_gauss_q, self.pre_gauss_sigma, self.pre_threshold_q, self.pre_threshold_val]:
            var.trace("w", self.preprocess)

        # Manual controls #############################################################################################
        manual_tab = ttk.Frame(self.control_panel)
        manual_tab.grid(row=0, column=0, sticky=tk.NSEW)
        self.fourier = True

        r = 0
        names = ["Error reduction", "Hybrid input-output", "Shrinkwrap", "Forward propagate (FFT)"]
        commands = [self.man_er, self.man_hio, self.man_sw, self.man_fft]
        self.ds_buttons = []
        for name, command in zip(names, commands):
            btn = ttk.Button(manual_tab, text=name, command=command)
            self.ds_buttons.append(btn)
            btn.grid(row=r, column=0, **btn_kwargs)
            r += 1

        ttk.Separator(manual_tab, orient="horizontal").grid(row=r, **sep_kwargs)
        r += 1

        names = ["Replace modulus", "Back propagate (IFFT)"]
        commands = [self.man_mod, self.man_ifft]
        self.fs_buttons = []
        for name, command in zip(names, commands):
            btn = ttk.Button(manual_tab, text=name, command=command)
            self.fs_buttons.append(btn)
            btn.grid(row=r, column=0, **btn_kwargs)
            r += 1

        # Automatic controls ##########################################################################################
        live_tab = ttk.Frame(self.control_panel)
        live_tab.grid(row=0, column=0, sticky=tk.NSEW)

        self.is_running = False

        self.start_button = ttk.Button(live_tab, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, rowspan=2, sticky=tk.NSEW, padx=2, pady=2)

        self.stop_button = ttk.Button(live_tab, text="Stop", command=self.stop)
        self.stop_button.grid(row=0, column=1, **btn_kwargs)

        self.erstop_button = ttk.Button(live_tab, text="Stop w/ ER", command=self.stop_with_er)
        self.erstop_button.grid(row=1, column=1, **btn_kwargs)

        ttk.Separator(live_tab, orient="horizontal").grid(row=2, **sep_kwargs)

        r = 3
        names = ["Re-center", "Remove twin", "Gaussian Blur", "Reset"]
        commands = [self.center, self.remove_twin, self.gaussian_blur, self.restart]
        self.auto_buttons = []
        for name, command in zip(names, commands):
            btn = ttk.Button(live_tab, text=name, command=command)
            self.auto_buttons.append(btn)
            btn.grid(row=r, column=0, columnspan=3, **btn_kwargs)
            r += 1

        self.save_msg = True
        ttk.Button(live_tab, text="Save results", command=self.save_result).grid(row=r, column=0, columnspan=3,
                                                                                 **btn_kwargs)

        # Parameter controls ##########################################################################################
        self.sw_sigma = tk.DoubleVar(value=2.0)
        self.sw_thresh = tk.DoubleVar(value=0.2)
        self.hio_beta = tk.DoubleVar(value=0.9)

        for tab in [manual_tab, live_tab]:
            params = ttk.LabelFrame(tab, text="Parameters", borderwidth=2)
            params.grid(row=20, column=0, columnspan=2, sticky=tk.S)

            ttk.Label(params, text="Shrinkwrap", justify=tk.CENTER).grid(row=0, column=0, columnspan=2)
            ttk.Separator(params, orient="vertical").grid(row=0, column=2, rowspan=4, sticky=tk.NS, padx=8, pady=8)
            ttk.Label(params, text="HIO", justify=tk.CENTER).grid(row=0, column=3)

            ttk.Label(params, text="Sigma", justify=tk.CENTER).grid(row=1, column=0)
            sw_sigma_input = ttk.Scale(params, from_=10.0, to=0.0, orient="vertical", variable=self.sw_sigma)
            sw_sigma_input.grid(row=2, column=0, **btn_kwargs)
            FormatLabel(params, textvariable=self.sw_sigma, format="{:.2f}").grid(row=3, column=0)

            ttk.Label(params, text="Thresh.", justify=tk.CENTER).grid(row=1, column=1)
            sw_thresh_input = ttk.Scale(params, from_=0.5, to=0.0, orient="vertical", variable=self.sw_thresh)
            sw_thresh_input.grid(row=2, column=1, **btn_kwargs)
            FormatLabel(params, textvariable=self.sw_thresh, format="{:.2f}").grid(row=3, column=1)

            ttk.Label(params, text="Beta", justify=tk.RIGHT).grid(row=1, column=3)
            hio_beta_input = ttk.Scale(params, from_=1.0, to=0.0, orient="vertical", variable=self.hio_beta)
            hio_beta_input.grid(row=2, column=3, **btn_kwargs)
            FormatLabel(params, textvariable=self.hio_beta, format="{:.2f}").grid(row=3, column=3)

        # Add the tabs to the control panel
        self.control_panel.add(data_tab, text="Data")
        self.control_panel.add(manual_tab, text="Manual")
        self.control_panel.add(live_tab, text="Auto")

        # Finally, make the object itself. Start with random shapes.
        impad = 2
        self.im_size = 500
        self.images = [np.abs(self.solver.fs_image), np.angle(self.solver.fs_image)]
        self.axes = []
        self.image_canvas = []
        self.scale_bars = []
        self.scale_bars_text = []

        for col, label, image, cmap in zip([0, 2], ["Amplitude", "Phase"], self.images, ["gray", "hsv"]):
            ttk.Label(self.root, text=label, font=("Arial", 20), justify=tk.CENTER).grid(row=0, column=col)
            fig = plt.figure(figsize=(1, 1), dpi=self.im_size)
            ax = fig.add_subplot(xticks=[], yticks=[])
            axim = ax.imshow(image, cmap=cmap, extent=(0, 1, 0, 1), interpolation_stage='rgba')
            fig.tight_layout(pad=0)
            cvs = FigureCanvasTkAgg(fig, master=self.root)
            cvs.get_tk_widget().grid(row=1, column=col, padx=impad, pady=impad)
            bar = Rectangle((0.04, 0.04), 0.35, 0.06, color='white')
            ax.add_patch(bar)
            bar_text = ax.text(0.05, 0.05, "Hello!", fontsize=4)
            for container, thing in zip([self.axes, self.image_canvas, self.scale_bars, self.scale_bars_text],
                                        [axim, cvs, bar, bar_text]):
                container.append(thing)

        self.control_panel.bind("<<NotebookTabChanged>>", self.update_images)

        self.clock = time.perf_counter()
        self.root.update()
        # showinfo("Welcome", "Welcome to Interactive CDI!\n\n"
        #                     "If you like this project, please give it a star on GitHub.")
        self.root.mainloop()

    def edit_det_params(self):
        for entry in self.det_params_entries:
            entry["state"] = "normal"
        self.det_params_edit_button["state"] = "disabled"
        self.det_params_lock_button["state"] = "normal"

    def lock_det_params(self):
        for entry in self.det_params_entries:
            entry["state"] = "disabled"
        self.det_params_edit_button["state"] = "normal"
        self.det_params_lock_button["state"] = "disabled"
        self.preprocess()
        self.update_images()

    def man_sw(self):
        self.solver.shrinkwrap(self.sw_sigma.get(), self.sw_thresh.get())
        self.axes[0].set(data=self.solver.support.array, clim=[0, 1])
        self.image_canvas[0].draw()

    def man_hio(self):
        self.solver.hio_constraint(self.hio_beta.get())
        self.update_images()

    def man_er(self):
        self.solver.er_constraint()
        self.update_images()

    def man_fft(self):
        self.fourier = True
        self.solver.fft()
        self.update_images()

    def man_ifft(self):
        self.fourier = False
        self.solver.ifft()
        self.update_images()

    def man_mod(self):
        self.solver.modulus_constraint()
        self.update_images()

    def start(self):
        self.is_running = True
        self.start_button.state(["disabled"])
        self.run()

    def stop(self):
        self.is_running = False
        self.start_button.state(["!disabled"])

    def stop_with_er(self):
        self.is_running = False
        self.start_button.state(["!disabled"])
        self.root.after(10, self.solver.er_iteration)
        self.root.after(15, self.update_images)

    def run(self):
        self.solver.hio_iteration(self.hio_beta.get())
        self.solver.shrinkwrap(self.sw_sigma.get(), self.sw_thresh.get())
        self.update_images()
        if self.is_running:
            self.root.after(10, self.run)

    def update_images(self, *_):
        pnl = self.control_panel.index("current")
        self.fourier = (pnl == DATA) or (pnl == MANUAL and self.fourier)
        if not pnl == AUTO:
            self.stop()
        if self.fourier:
            self.images = [np.sqrt(np.abs(self.solver.fs_image)), np.angle(self.solver.fs_image)]
            for button in self.ds_buttons:
                button.state(["disabled"])
            for button in self.fs_buttons:
                button.state(["!disabled"])
        else:
            self.images = [np.abs(self.solver.ds_image), np.angle(self.solver.ds_image)]
            for button in self.ds_buttons:
                button.state(["!disabled"])
            for button in self.fs_buttons:
                button.state(["disabled"])
        clims = [(0, self.images[0].max()), (-np.pi, np.pi)]
        width, text = self.auto_scale_bar()
        for img, ax, canvas, clim, bar, bar_text in zip(self.images, self.axes, self.image_canvas, clims,
                                                        self.scale_bars, self.scale_bars_text):
            bar.set(width=width)
            bar_text.set(text=text)
            ax.set(data=img, clim=clim)
            canvas.draw()

    def auto_scale_bar(self):
        if self.fourier:
            pixel_size = self.det_pitch.get() * 10**-6
            if self.pre_bin_q.get():
                pixel_size *= self.pre_bin_factor.get()
        else:
            pixel_size = self.solver.pixel_size
        number, power = ut.sig_round(self.solver.imsize * 0.35 * pixel_size)
        width = number * 10 ** power / (pixel_size * self.solver.imsize)
        text = f"{number}{'0'*(power%3)} {UNITS[power//3]}"
        return width, text

    def center(self):
        self.solver.center()
        self.update_images()

    def remove_twin(self):
        self.solver.remove_twin()
        self.update_images()

    def gaussian_blur(self):
        self.solver.gaussian_blur()
        self.update_images()

    def load_data(self):
        self.data.load_data()
        self.preprocess()
        self.restart()

    def load_bkgd(self):
        self.data.load_bkgd()
        self.preprocess()
        self.restart()

    def preprocess(self, *_):
        self.solver = phasing.Solver(self.data.preprocess(self.pre_bkgd.get(),
                                                          self.pre_bin_q.get(),
                                                          self.pre_bin_factor.get(),
                                                          self.pre_crop_q.get(),
                                                          self.pre_crop_factor.get(),
                                                          self.pre_gauss_q.get(),
                                                          self.pre_gauss_sigma.get(),
                                                          self.pre_threshold_q.get(),
                                                          self.pre_threshold_val.get(),
                                                          )
                                     )
        try:
            if self.pre_bin_q.get():
                det_pitch = self.det_pitch.get() * self.pre_bin_factor.get()
            else:
                det_pitch = self.det_pitch.get()
            self.solver.set_scale(det_pitch, self.det_dist.get(), self.wavelength.get())
        except tk.TclError:
            self.solver.pixel_size = None
        self.update_images()

    def save_result(self):
        if self.save_msg:
            # Show this message the first time only.
            showinfo("Info", "Because there are multiple files to save, you will be asked to input a folder rather "
                             "than simply a file name. It is highly recommended that you create a new folder for "
                             "these files. If you select a folder that already contains output from this app, "
                             "the old files WILL be overwritten!")
            self.save_msg = False
        save_dir = askdirectory()
        np.save(f"{save_dir}/ds_raw.npy", self.solver.ds_image)
        for img, space in zip([self.solver.ds_image, self.solver.fs_image], ["ds", "fs"]):
            plt.imsave(f"{save_dir}/{space}_amplitude.png", np.abs(img), cmap="gray")
            plt.imsave(f"{save_dir}/{space}_phase.png", np.angle(img), cmap="hsv")
            plt.imsave(f"{save_dir}/{space}_combined.png", ut.complex_composite_image(img, dark_background=True))

    def restart(self):
        self.hio_beta.set(0.9)
        self.sw_sigma.set(2.0)
        self.sw_thresh.set(0.2)
        self.solver.reset()
        self.update_images()


class FormatLabel(tk.Label):

    def __init__(self, master=None, cnf=None, **kw):

        # default values
        if cnf is None:
            cnf = {}
        self._format = '{}'
        self._textvariable = None

        # get new format and remove it from `kw` so later `super().__init__` doesn't use them
        # (it would get error message)
        if 'format' in kw:
            self._format = kw['format']
            del kw['format']

        # get `textvariable` to assign own function which set formatted text in Label when variable change value
        if 'textvariable' in kw:
            self._textvariable = kw['textvariable']
            self._textvariable.trace('w', self._update_text)
            del kw['textvariable']

        # run `Label.__init__` without `format` and `textvariable`
        super().__init__(master, cnf={}, **kw)

        # update text after running `Label.__init__`
        if self._textvariable:
            self._update_text(self._textvariable, '', 'w')

    def _update_text(self, a, b, c):
        """update text in label when variable change value"""
        self["text"] = self._format.format(self._textvariable.get())


if __name__ == "__main__":
    App()
