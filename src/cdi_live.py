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
from matplotlib.pyplot import imsave

import src.phasing as phasing
import src.diffraction as diffraction
import src.utils as ut


class App:
    def __init__(self):
        self.data = diffraction.LoadData()
        self.solver = phasing.Solver(self.data.preprocess())

        self.root = tk.Tk()
        self.root.title("Interactive CDI (v0.5)")

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
        ttk.Separator(data_tab, orient="horizontal").grid(row=r, **sep_kwargs)
        r += 1
        ttk.Label(data_tab, text="Image processing", font=('Arial', 12, 'underline')).grid(row=r, **sep_kwargs)
        r += 1
        self.pre_bkgd = tk.BooleanVar(value=False)
        ttk.Checkbutton(data_tab, text="Subtract background", variable=self.pre_bkgd).grid(row=r, column=0,
                                                                                           columnspan=3, **btn_kwargs)
        r += 1
        self.pre_binning = tk.BooleanVar(value=False)
        self.pre_binfact = tk.IntVar(value=1)
        ttk.Checkbutton(data_tab, text="Bin pixels", variable=self.pre_binning).grid(row=r, column=0, **btn_kwargs)
        FormatLabel(data_tab, textvariable=self.pre_binfact, format="{:.0f}").grid(row=r, column=1, **btn_kwargs)
        r += 1
        ttk.Scale(data_tab, from_=1, to=10, orient="horizontal", variable=self.pre_binfact).grid(row=r, column=0,
                                                                                                 columnspan=2,
                                                                                                 **btn_kwargs)

        r += 1
        self.pre_gauss = tk.BooleanVar(value=False)
        self.pre_sigma = tk.DoubleVar(value=0.0)
        ttk.Checkbutton(data_tab, text="Gaussian blur", variable=self.pre_gauss).grid(row=r, column=0, **btn_kwargs)
        FormatLabel(data_tab, textvariable=self.pre_sigma, format="{:.2f}").grid(row=r, column=1, **btn_kwargs)
        r += 1
        ttk.Scale(data_tab, from_=0.0, to=5.0, orient="horizontal", variable=self.pre_sigma).grid(row=r, column=0,
                                                                                                  columnspan=2,
                                                                                                  **btn_kwargs)

        r += 1
        self.pre_thresholding = tk.BooleanVar(value=False)
        self.pre_thresh = tk.DoubleVar(value=0.0)
        ttk.Checkbutton(data_tab, text="Threshold", variable=self.pre_thresholding).grid(row=r, column=0,
                                                                                         **btn_kwargs)
        FormatLabel(data_tab, textvariable=self.pre_thresh, format="{:.2f}").grid(row=r, column=1, **btn_kwargs)
        r += 1
        ttk.Scale(data_tab, from_=0.0, to=1.0, orient="horizontal", variable=self.pre_thresh).grid(row=r, column=0,
                                                                                                   columnspan=2,
                                                                                                   **btn_kwargs)

        r += 1
        self.pre_vignette = tk.BooleanVar(value=False)
        self.pre_vsigma = tk.DoubleVar(value=3)
        ttk.Checkbutton(data_tab, text="Vignette", variable=self.pre_vignette).grid(row=r, column=0, **btn_kwargs)
        FormatLabel(data_tab, textvariable=self.pre_vsigma, format="{:.2f}").grid(row=r, column=1, **btn_kwargs)
        r += 1
        ttk.Scale(data_tab, from_=3, to=0.1, orient="horizontal", variable=self.pre_vsigma).grid(row=r, column=0,
                                                                                                 columnspan=2,
                                                                                                 **btn_kwargs)

        for var in [self.pre_bkgd, self.pre_binning, self.pre_binfact, self.pre_gauss, self.pre_sigma,
                    self.pre_thresholding, self.pre_thresh, self.pre_vignette, self.pre_vsigma]:
            var.trace("w", self.preprocess)

        # Manual controls #############################################################################################
        manual_tab = ttk.Frame(self.control_panel)
        manual_tab.grid(row=0, column=0, sticky=tk.NSEW)
        self.fourier = True

        r = 0
        names = ["Shrinkwrap", "Hybrid input-output", "Error reduction", "Forward propagate (FFT)"]
        commands = [self.man_sw, self.man_hio, self.man_er, self.man_fft]
        self.ds_buttons = []
        for name, command in zip(names, commands):
            btn = ttk.Button(manual_tab, text=name, command=command)
            self.ds_buttons.append(btn)
            btn.grid(row=r, column=0, **btn_kwargs)
            r += 1

        ttk.Separator(manual_tab, orient="horizontal").grid(row=r, **sep_kwargs)
        r += 1

        names = ["Back propagate (IFFT)", "Replace modulus"]
        commands = [self.man_ifft, self.man_mod]
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
        self.im_size = 450
        self.img_left = ut.amp_to_photo_image(np.abs(self.solver.ds_image), size=self.im_size)
        self.img_right = ut.phase_to_photo_image(self.solver.ds_image, size=self.im_size)

        self.label_left = ttk.Label(self.root, text="Amplitude", font=("Arial", 20), justify=tk.CENTER)
        self.label_left.grid(row=0, column=0)
        self.disp_left = ttk.Label(self.root, image=self.img_left)
        self.disp_left.grid(row=1, column=0, padx=impad, pady=impad)

        self.label_right = ttk.Label(self.root, text="Phase", font=("Arial", 20), justify=tk.CENTER)
        self.label_right.grid(row=0, column=2)
        self.disp_right = ttk.Label(self.root, image=self.img_right)
        self.disp_right.grid(row=1, column=2, padx=impad, pady=impad)

        self.control_panel.bind("<<NotebookTabChanged>>", self.update_images)

        self.clock = time.perf_counter()
        self.root.mainloop()

    def man_sw(self):
        self.solver.shrinkwrap(self.sw_sigma.get(), self.sw_thresh.get())
        self.img_left = ut.amp_to_photo_image(np.uint8(self.solver.support.array), size=self.im_size)
        self.disp_left.configure(image=self.img_left)
        self.disp_left.image = self.img_left

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
        i = self.control_panel.index("current")
        self.fourier = i == 0 or (i == 1 and self.fourier)
        if not i == 2:
            self.stop()
        if self.fourier:
            self.img_left = ut.amp_to_photo_image(np.sqrt(np.abs(self.solver.fs_image)), size=self.im_size)
            self.img_right = ut.phase_to_photo_image(self.solver.fs_image, size=self.im_size)
            for button in self.ds_buttons:
                button.state(["disabled"])
            for button in self.fs_buttons:
                button.state(["!disabled"])
        else:
            self.img_left = ut.amp_to_photo_image(np.abs(self.solver.ds_image), mask=self.solver.support.array,
                                                  size=self.im_size)
            self.img_right = ut.phase_to_photo_image(self.solver.ds_image, size=self.im_size)
            for button in self.ds_buttons:
                button.state(["!disabled"])
            for button in self.fs_buttons:
                button.state(["disabled"])
        self.disp_left.configure(image=self.img_left)
        self.disp_left.image = self.img_left
        self.disp_right.configure(image=self.img_right)
        self.disp_right.image = self.img_right

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
                                                          self.pre_binning.get(),
                                                          self.pre_binfact.get(),
                                                          self.pre_gauss.get(),
                                                          self.pre_sigma.get(),
                                                          self.pre_thresholding.get(),
                                                          self.pre_thresh.get(),
                                                          self.pre_vignette.get(),
                                                          self.pre_vsigma.get()
                                                          )
                                     )
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
        imsave(f"{save_dir}/ds_amplitude.png", np.abs(self.solver.ds_image), cmap="gray")
        imsave(f"{save_dir}/ds_phase.png", np.angle(self.solver.ds_image), cmap="hsv")
        imsave(f"{save_dir}/ds_combined.png", ut.complex_composite_image(self.solver.ds_image, dark_background=True))
        imsave(f"{save_dir}/fs_amplitude.png", np.abs(self.solver.fs_image), cmap="gray")
        imsave(f"{save_dir}/fs_phase.png", np.angle(self.solver.fs_image), cmap="hsv")
        imsave(f"{save_dir}/fs_combined.png", ut.complex_composite_image(self.solver.fs_image, dark_background=True))

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
