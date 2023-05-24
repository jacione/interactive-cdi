"""
Live phase retrieval GUI.

NOTES:
    - The main looping feature uses alternating HIO and shrinkwrap algorithms.
    - There are some pathologically difficult objects, and I don't know why. I'll try to keep a record of which seeds
    don't reconstruct well:
        | seed      | shapes    |
        | 345       | 5         |
"""


import time
import tkinter as tk
from tkinter.messagebox import showerror
from tkinter.filedialog import askopenfilename, asksaveasfilename
import tkinter.ttk as ttk
from random import randint

import numpy as np
from matplotlib.pyplot import imread, imsave

import src.phasing as phasing
import src.sample as sample
import src.utils as ut


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Phase retrieval stepper")

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        self.control_panel = ttk.Notebook(self.root)
        self.control_panel.grid(row=1, column=1, sticky=tk.NSEW)

        btn_kwargs = {"sticky": tk.EW, "padx": 2, "pady": 2}
        sep_kwargs = {"column": 0, "columnspan": 3, "sticky": tk.EW, "pady": 3}

        # Data controls ###############################################################################################
        data_tab = ttk.Frame(self.control_panel)
        r = 0
        data_tab.grid(row=r, column=0, stick=tk.EW)

        r += 1
        ttk.Label(data_tab, text="Seed (int): ").grid(row=r, column=0, sticky=tk.E)
        self.sim_seed = tk.IntVar(value=randint(0, 9999999))
        ttk.Entry(data_tab, textvariable=self.sim_seed, width=10).grid(row=r, column=1, **btn_kwargs)

        r += 1
        ttk.Label(data_tab, text="# of shapes: ").grid(row=r, column=0, sticky=tk.E)
        self.sim_nshapes = tk.IntVar(value=5)
        ttk.Entry(data_tab, textvariable=self.sim_nshapes, width=10).grid(row=r, column=1, **btn_kwargs)

        r += 1
        ttk.Label(data_tab, text="Camera: ").grid(row=r, column=0, sticky=tk.E)
        self.sim_bits = tk.StringVar(value="N/A")
        self.bits_dict = {"Ideal": None} | {f"{b}-bit": (2**b) - 1 for b in [8, 10, 12, 14, 16, 20, 24]}
        ttk.OptionMenu(data_tab, self.sim_bits, "Ideal", *self.bits_dict.keys()).grid(row=r, column=1, **btn_kwargs)

        r += 1
        ttk.Label(data_tab, text="Saturation: ").grid(row=r, column=0, sticky=tk.E)
        self.sim_sat = tk.StringVar(data_tab, value="0%")
        self.sat_dict = {f"{p}%": 1 + (p / 100) for p in [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]}
        ttk.OptionMenu(data_tab, self.sim_sat, "0%", *self.sat_dict.keys()).grid(row=r, column=1, **btn_kwargs)

        r += 1
        ttk.Label(data_tab, text="Summed images: ").grid(row=r, column=0, sticky=tk.E)
        self.sim_acc = tk.IntVar(data_tab, value=1)
        ttk.Spinbox(data_tab, from_=1, to=100, textvariable=self.sim_acc, width=5).grid(row=r, column=1, **btn_kwargs)

        r += 1
        ttk.Label(data_tab, text="Angle: ").grid(row=r, column=0, sticky=tk.E)
        self.sim_rot = tk.StringVar(data_tab, value="0%")
        self.rot_dict = {f"{p} deg": p for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]}
        ttk.OptionMenu(data_tab, self.sim_rot, "0 deg", *self.rot_dict.keys()).grid(row=r, column=1, **btn_kwargs)

        r += 1
        ttk.Label(data_tab, text="Interp. mode: ").grid(row=r, column=0, sticky=tk.E)
        self.sim_spl = tk.StringVar(data_tab, value="0%")
        self.spl_dict = {"linear": 1, "quadratic": 2, "cubic": 3, "quartic": 4, "quintic": 5}
        ttk.OptionMenu(data_tab, self.sim_spl, "cubic", *self.spl_dict.keys()).grid(row=r, column=1, **btn_kwargs)

        for var in [self.sim_seed, self.sim_nshapes, self.sim_bits, self.sim_sat, self.sim_acc, self.sim_rot,
                    self.sim_spl]:
            var.trace("w", self.generate)

        r += 1
        ttk.Button(data_tab, text="Generate data", command=self.generate).grid(row=r, column=0, columnspan=2,
                                                                               **btn_kwargs)
        r += 1
        ttk.Button(data_tab, text="Show", command=self.show_sample).grid(row=r, column=0, columnspan=2, **btn_kwargs)

        r += 1
        ttk.Separator(data_tab, orient="horizontal").grid(row=r, **sep_kwargs)
        r += 1
        ttk.Button(data_tab, text="Load diffraction", command=self.load_data).grid(row=r, column=0, columnspan=2,
                                                                                   **btn_kwargs)
        r += 1
        ttk.Separator(data_tab, orient="horizontal").grid(row=r, **sep_kwargs)
        r += 1
        ttk.Button(data_tab, text="Save result", command=self.save_result).grid(row=r, column=0, columnspan=2,
                                                                                **btn_kwargs)

        # Create the object itself based on these parameters.
        self.simulated = True
        self.is_running = False
        self.sim_size = 400
        self.sample = sample.RandomShapes(self.sim_size, self.sim_seed.get(), self.sim_nshapes.get())
        self.solver = phasing.Solver(self.sample.detect(self.sim_acc.get(),
                                                        self.sat_dict[self.sim_sat.get()],
                                                        self.bits_dict[self.sim_bits.get()],
                                                        self.rot_dict[self.sim_rot.get()],
                                                        self.spl_dict[self.sim_spl.get()]))

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

        self.start_button = ttk.Button(live_tab, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, rowspan=2, sticky=tk.NSEW, padx=2, pady=2)

        self.stop_button = ttk.Button(live_tab, text="Stop", command=self.start)
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
        self.img_left = ut.amp_to_photo_image(np.abs(self.solver.ds_image))
        self.img_right = ut.phase_to_photo_image(np.angle(self.solver.ds_image))

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
        self.img_left = ut.amp_to_photo_image(np.uint8(self.solver.support.array))
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
            self.img_left = ut.amp_to_photo_image(np.sqrt(np.abs(self.solver.fs_image)))
            self.img_right = ut.phase_to_photo_image(np.angle(self.solver.fs_image))
            for button in self.ds_buttons:
                button.state(["disabled"])
            for button in self.fs_buttons:
                button.state(["!disabled"])
        else:
            self.img_left = ut.amp_to_photo_image(np.abs(self.solver.ds_image), mask=self.solver.support.array)
            self.img_right = ut.phase_to_photo_image(np.angle(self.solver.ds_image))
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

    def generate(self, *_):
        try:
            self.simulated = True
            self.sample = sample.RandomShapes(self.sim_size, self.sim_seed.get(), self.sim_nshapes.get())
            self.solver.diffraction = self.sample.detect(self.sim_acc.get(),
                                                         self.sat_dict[self.sim_sat.get()],
                                                         self.bits_dict[self.sim_bits.get()],
                                                         self.rot_dict[self.sim_rot.get()],
                                                         self.spl_dict[self.sim_spl.get()])
            self.restart()
        except tk.TclError:
            return

    def show_sample(self):
        if self.simulated:
            self.sample.show()
        else:
            showerror("Error", "Ground truth knowledge only exists for simulated data.")

    def load_data(self):
        self.simulated = False
        f = askopenfilename()
        self.solver.diffraction = imread(f)
        self.restart()

    def save_result(self):
        imsave(asksaveasfilename(defaultextension="png"), np.abs(self.solver.ds_image))

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

        # get new format and remove it from `kw` so later `super().__init__` doesn't use them (it would get error message)
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
