import time
import tkinter as tk
from tkinter.messagebox import showerror
from tkinter.simpledialog import askinteger
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
        # GUI starts here
        self.root = tk.Tk()
        self.root.title("Phase retrieval stepper")

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        control_panel = ttk.Notebook(self.root)
        control_panel.grid(row=1, column=1, sticky=tk.NSEW)

        btn_kwargs = {"sticky": tk.EW, "padx": 2, "pady": 2}
        sep_kwargs = {"column": 0, "columnspan": 3, "sticky": tk.EW, "pady": 3}

        # Automatic controls ##########################################################################################
        auto_tab = ttk.Frame(control_panel)
        auto_tab.grid(row=0, column=0, sticky=tk.EW)

        # Create automatic control widgets
        auto_hio_button = ttk.Button(auto_tab, text="HIO", command=self.run_hio)
        auto_hio_button.grid(row=0, column=0, **btn_kwargs)
        ttk.Label(auto_tab, text="x").grid(row=0, column=1, **btn_kwargs)
        self.num_hio = tk.IntVar(value=85)
        num_hio_input = ttk.Entry(auto_tab, textvariable=self.num_hio, width=5)
        num_hio_input.grid(row=0, column=2, **btn_kwargs)

        auto_er_button = ttk.Button(auto_tab, text="ER", command=self.run_er)
        auto_er_button.grid(row=1, column=0, **btn_kwargs)
        self.num_er = tk.IntVar(value=15)
        ttk.Label(auto_tab, text="x").grid(row=1, column=1, **btn_kwargs)
        num_er_input = ttk.Entry(auto_tab, textvariable=self.num_er, width=5)
        num_er_input.grid(row=1, column=2, **btn_kwargs)

        self.progbar = ttk.Progressbar(auto_tab, orient='horizontal', mode="determinate")
        self.progbar.grid(row=2, column=0, columnspan=3, **btn_kwargs)

        auto_sw_button = ttk.Button(auto_tab, text="Shrinkwrap", command=self.do_sw)
        auto_sw_button.grid(row=3, column=0, columnspan=3, **btn_kwargs)

        auto_twin_button = ttk.Button(auto_tab, text="Remove twin", command=self.remove_twin)
        auto_twin_button.grid(row=4, column=0, columnspan=3, **btn_kwargs)

        auto_blur_button = ttk.Button(auto_tab, text="Blur", command=self.gaussian_blur)
        auto_blur_button.grid(row=5, column=0, columnspan=3, **btn_kwargs)

        auto_reset_button = ttk.Button(auto_tab, text="Reset", command=self.restart)
        auto_reset_button.grid(row=6, column=0, columnspan=3, **btn_kwargs)

        self.auto_buttons = [auto_er_button, auto_hio_button, auto_sw_button, auto_twin_button, auto_blur_button,
                             auto_reset_button]

        # Manual controls #############################################################################################
        manual_tab = ttk.Frame(control_panel)
        manual_tab.grid(row=0, column=0, sticky=tk.EW)

        # Create manual control widgets
        sw_button = ttk.Button(manual_tab, text="Shrinkwrap", command=self.do_sw)
        sw_button.grid(row=0, column=0, **btn_kwargs)

        hio_button = ttk.Button(manual_tab, text="HIO constraint", command=self.do_hio)
        hio_button.grid(row=1, column=0,  **btn_kwargs)

        er_button = ttk.Button(manual_tab, text="ER constraint", command=self.do_er)
        er_button.grid(row=2, column=0, **btn_kwargs)

        fft_button = ttk.Button(manual_tab, text="FFT", command=self.to_fourier)
        fft_button.grid(row=3, column=0, **btn_kwargs)

        ttk.Separator(manual_tab, orient="horizontal").grid(row=4, **sep_kwargs)

        ifft_button = ttk.Button(manual_tab, text="IFFT", command=self.to_direct)
        ifft_button.grid(row=5, column=0, **btn_kwargs)

        modulus_button = ttk.Button(manual_tab, text="Modulus constraint", command=self.do_modulus)
        modulus_button.grid(row=6, column=0, **btn_kwargs)

        # Group buttons
        self.ds_buttons = [er_button, sw_button, fft_button, hio_button]
        self.fs_buttons = [ifft_button, modulus_button]

        # Data controls ###############################################################################################
        data_tab = ttk.Frame(control_panel)
        data_tab.grid(row=0, column=0, stick=tk.EW)

        ttk.Label(data_tab, text="Seed (int): ").grid(row=1, column=0, sticky=tk.E)
        self.sim_seed = tk.IntVar(value=randint(0, 9999999))
        ttk.Entry(data_tab, textvariable=self.sim_seed, width=10).grid(row=1, column=1, **btn_kwargs)

        ttk.Label(data_tab, text="Camera: ").grid(row=2, column=0, sticky=tk.E)
        self.sim_bits = tk.StringVar(value="N/A")
        self.bits_dict = {"Ideal": None} | {f"{b}-bit": (2**b) - 1 for b in [8, 10, 12, 14, 16, 24]}
        ttk.OptionMenu(data_tab, self.sim_bits, "Ideal", *self.bits_dict.keys()).grid(row=2, column=1, **btn_kwargs)

        ttk.Label(data_tab, text="Saturation: ").grid(row=3, column=0, sticky=tk.E)
        self.sim_sat = tk.StringVar(data_tab, value="0%")
        self.sat_dict = {f"{p}%": 1 + (p / 100) for p in [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]}
        ttk.OptionMenu(data_tab, self.sim_sat, "0%", *self.sat_dict.keys()).grid(row=3, column=1, **btn_kwargs)

        ttk.Button(data_tab, text="Generate data", command=self.generate).grid(row=4, column=0, columnspan=2,
                                                                               **btn_kwargs)
        ttk.Button(data_tab, text="Show", command=self.show_sample).grid(row=5, column=0, columnspan=2, **btn_kwargs)

        ttk.Separator(data_tab, orient="horizontal").grid(row=6, **sep_kwargs)
        ttk.Button(data_tab, text="Load diffraction", command=self.load_data).grid(row=7, column=0, columnspan=2,
                                                                                   **btn_kwargs)
        ttk.Separator(data_tab, orient="horizontal").grid(row=8, **sep_kwargs)
        ttk.Button(data_tab, text="Save result", command=self.save_result).grid(row=9, column=0, columnspan=2,
                                                                                **btn_kwargs)

        # Add the tabs to the control panel
        control_panel.add(auto_tab, text="Auto")
        control_panel.add(manual_tab, text="Manual")
        control_panel.add(data_tab, text="Data")

        # Parameter controls ##########################################################################################
        params = ttk.LabelFrame(self.root, text="Parameters", borderwidth=2)
        params.grid(row=2, column=1, sticky=tk.S)

        ttk.Label(params, text="Shrinkwrap", justify=tk.CENTER).grid(row=0, column=0, columnspan=2)

        ttk.Label(params, text="Sigma:", justify=tk.RIGHT).grid(row=1, column=0, sticky=tk.E)
        self.sw_sigma = tk.DoubleVar(value=2.0)
        sw_sigma_input = ttk.Entry(params, textvariable=self.sw_sigma, width=6)
        sw_sigma_input.grid(row=1, column=1, **btn_kwargs)

        ttk.Label(params, text="Threshold:", justify=tk.RIGHT).grid(row=2, column=0, sticky=tk.E)
        self.sw_thresh = tk.DoubleVar(value=0.1)
        sw_thresh_input = ttk.Entry(params, textvariable=self.sw_thresh, width=6)
        sw_thresh_input.grid(row=2, column=1, **btn_kwargs)

        ttk.Separator(params, orient="horizontal").grid(row=3, **sep_kwargs)

        ttk.Label(params, text="HIO", justify=tk.CENTER).grid(row=4, column=0, columnspan=2)

        ttk.Label(params, text="Beta:", justify=tk.RIGHT).grid(row=5, column=0, sticky=tk.E)
        self.hio_beta = tk.DoubleVar(value=0.9)
        hio_beta_input = ttk.Entry(params, textvariable=self.hio_beta, width=6)
        hio_beta_input.grid(row=5, column=1, **btn_kwargs)

        ttk.Separator(params, orient="horizontal").grid(row=6, **sep_kwargs)

        # Finally, make the object itself. Start with random shapes.
        impad = 2
        self.simulated = True
        self.fourier = True
        self.sim_size = 400
        self.sample = sample.RandomShapes(self.sim_size, self.sim_seed.get())
        self.solver = phasing.Solver(self.sample.detect())

        self.img_left = ut.amp_to_photo_image(np.sqrt(np.abs(self.solver.fs_image)))
        self.img_right = ut.phase_to_photo_image(np.angle(self.solver.fs_image))

        self.label_left = ttk.Label(self.root, text="Amplitude", font=("Arial", 20), justify=tk.CENTER)
        self.label_left.grid(row=0, column=0)
        self.disp_left = ttk.Label(self.root, image=self.img_left)
        self.disp_left.grid(row=1, column=0, rowspan=2, padx=impad, pady=impad)

        self.label_right = ttk.Label(self.root, text="Phase", font=("Arial", 20), justify=tk.CENTER)
        self.label_right.grid(row=0, column=2)
        self.disp_right = ttk.Label(self.root, image=self.img_right)
        self.disp_right.grid(row=1, column=2, rowspan=2, padx=impad, pady=impad)

        self.to_fourier()
        self.clock = time.perf_counter()
        self.root.mainloop()

    def update_images(self):
        if self.fourier:
            self.img_left = ut.amp_to_photo_image(np.sqrt(np.abs(self.solver.fs_image)))
            self.img_right = ut.phase_to_photo_image(np.angle(self.solver.fs_image))
        else:
            self.img_left = ut.amp_to_photo_image(np.abs(self.solver.ds_image))
            self.img_right = ut.phase_to_photo_image(np.angle(self.solver.ds_image))
        self.disp_left.configure(image=self.img_left)
        self.disp_left.image = self.img_left
        self.disp_right.configure(image=self.img_right)
        self.disp_right.image = self.img_right

    def do_er(self):
        self.solver.er_constraint()
        self.update_images()
        pass

    def do_hio(self):
        if 0 < self.hio_beta.get() < 1:
            self.solver.hio_constraint(self.hio_beta.get())
        else:
            showerror("Error", "HIO failed one or more checks:\n0 < beta < 1")
            return -1
        self.update_images()

    def do_sw(self):
        if (0 <= self.sw_sigma.get() <= 20) and (0 < self.sw_thresh.get() < 1):
            self.solver.shrinkwrap(self.sw_sigma.get(), self.sw_thresh.get())
            self.img_left = ut.amp_to_photo_image(np.uint8(self.solver.support.array))
            self.disp_left.configure(image=self.img_left)
            self.disp_left.image = self.img_left
        else:
            showerror("Error", "Shrinkwrap failed one or more checks:\n0 < sigma < 20\n0 < threshold < 1")
            return -1

    def do_modulus(self):
        self.solver.modulus_constraint()
        self.update_images()
        pass

    def to_fourier(self):
        self.fourier = True
        for button in self.ds_buttons:
            button.state(["disabled"])
        for button in self.fs_buttons:
            button.state(["!disabled"])
        self.solver.fft()
        self.update_images()
        pass

    def to_direct(self):
        self.fourier = False
        for button in self.ds_buttons:
            button.state(["!disabled"])
        for button in self.fs_buttons:
            button.state(["disabled"])
        self.solver.ifft()
        self.update_images()
        pass

    def iterate_er(self, i):
        self.solver.er_iteration()
        self.update_images()
        self.progbar["value"] = 100*(1 - i/self.num_er.get())
        if i > 0:
            self.root.after(10, self.iterate_er, i-1)
        else:
            for button in self.auto_buttons + self.ds_buttons:
                button.state(["!disabled"])
            self.progbar["value"] = 0

    def run_er(self):
        if self.fourier:
            self.to_direct()
        for button in self.auto_buttons + self.ds_buttons + self.fs_buttons:
            button.state(["disabled"])
        self.iterate_er(self.num_er.get())

    def iterate_hio(self, beta, i):
        self.solver.hio_iteration(beta)
        self.update_images()
        self.progbar["value"] = 100*(1 - i/self.num_hio.get())
        if i > 0:
            self.root.after(10, self.iterate_hio, beta, i-1)
        else:
            for button in self.auto_buttons + self.ds_buttons:
                button.state(["!disabled"])
            self.progbar["value"] = 0

    def run_hio(self):
        if self.fourier:
            self.to_direct()
        if not (0 < self.hio_beta.get() < 1):
            showerror("Error", "HIO failed one or more checks:\n0 < beta < 1")
            return
        for button in self.auto_buttons + self.ds_buttons + self.fs_buttons:
            button.state(["disabled"])
        self.iterate_hio(self.hio_beta.get(), self.num_hio.get())

    def remove_twin(self):
        self.solver.remove_twin()
        self.update_images()

    def gaussian_blur(self):
        self.solver.gaussian_blur()
        self.update_images()

    def generate(self):
        self.simulated = True
        self.sample = sample.RandomShapes(self.sim_size, self.sim_seed.get())
        self.solver.diffraction = self.sample.detect(self.sat_dict[self.sim_sat.get()],
                                                     self.bits_dict[self.sim_bits.get()])
        self.to_fourier()
        self.restart()

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
        self.solver.reset()
        self.update_images()


if __name__ == "__main__":
    App()
