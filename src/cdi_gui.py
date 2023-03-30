import time
import tkinter as tk
from tkinter.messagebox import showerror
import tkinter.ttk as ttk

import numpy as np

import src.phasing as phasing
import src.utils as ut


class App:
    def __init__(self, imsize=200):
        self.root = tk.Tk()
        self.root.title("Phase retrieval stepper")

        impad = 2

        self.solver = phasing.Solver(size=imsize)
        self.fourier = True
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

        self.controls = ttk.Notebook(self.root)
        self.controls.grid(row=1, column=1, sticky=tk.NSEW)

        btn_kwargs = {"sticky": tk.EW, "padx": 2, "pady": 2}
        sep_kwargs = {"column": 0, "columnspan": 3, "sticky": tk.EW, "pady": 3}

        # Manual controls #############################################################################################
        self.manual = ttk.Frame(self.controls)
        self.manual.grid(row=0, column=0, sticky=tk.EW)

        # Create manual control widgets
        self.sw_button = ttk.Button(self.manual, text="Shrinkwrap", command=self.do_sw)
        self.sw_button.grid(row=0, column=0, **btn_kwargs)

        self.hio_button = ttk.Button(self.manual, text="HIO constraint", command=self.do_hio)
        self.hio_button.grid(row=1, column=0,  **btn_kwargs)

        self.er_button = ttk.Button(self.manual, text="ER constraint", command=self.do_er)
        self.er_button.grid(row=2, column=0, **btn_kwargs)

        self.fft_button = ttk.Button(self.manual, text="FFT", command=self.to_fourier)
        self.fft_button.grid(row=3, column=0, **btn_kwargs)

        ttk.Separator(self.manual, orient="horizontal").grid(row=4, **sep_kwargs)

        self.ifft_button = ttk.Button(self.manual, text="IFFT", command=self.to_direct)
        self.ifft_button.grid(row=5, column=0, **btn_kwargs)

        self.modulus_button = ttk.Button(self.manual, text="Mod constraint", command=self.do_modulus)
        self.modulus_button.grid(row=6, column=0, **btn_kwargs)

        # Group buttons
        self.ds_buttons = [self.er_button, self.sw_button, self.fft_button, self.hio_button]
        self.fs_buttons = [self.ifft_button, self.modulus_button]

        # Automatic controls ##########################################################################################
        self.auto = ttk.Frame(self.controls)
        self.auto.grid(row=0, column=0, sticky=tk.EW)

        # Create automatic control widgets
        self.auto_hio_button = ttk.Button(self.auto, text="HIO", command=self.run_hio)
        self.auto_hio_button.grid(row=0, column=0, **btn_kwargs)
        ttk.Label(self.auto, text="x").grid(row=0, column=1, **btn_kwargs)
        self.num_hio = tk.IntVar(value=50)
        self.num_hio_input = ttk.Entry(self.auto, textvariable=self.num_hio, width=5)
        self.num_hio_input.grid(row=0, column=2, **btn_kwargs)

        self.auto_er_button = ttk.Button(self.auto, text="ER", command=self.run_er)
        self.auto_er_button.grid(row=1, column=0, **btn_kwargs)
        self.num_er = tk.IntVar(value=5)
        ttk.Label(self.auto, text="x").grid(row=1, column=1, **btn_kwargs)
        self.num_er_input = ttk.Entry(self.auto, textvariable=self.num_er, width=5)
        self.num_er_input.grid(row=1, column=2, **btn_kwargs)

        self.progbar = ttk.Progressbar(self.auto, orient='horizontal', mode="determinate")
        self.progbar.grid(row=2, column=0, columnspan=3, **btn_kwargs)

        self.auto_sw_button = ttk.Button(self.auto, text="Shrinkwrap", command=self.do_sw)
        self.auto_sw_button.grid(row=3, column=0, columnspan=3, **btn_kwargs)

        self.auto_twin_button = ttk.Button(self.auto, text="Remove twin", command=self.remove_twin)
        self.auto_twin_button.grid(row=4, column=0, columnspan=3, **btn_kwargs)

        self.auto_blur_button = ttk.Button(self.auto, text="Blur", command=self.gaussian_blur)
        self.auto_blur_button.grid(row=5, column=0, columnspan=3, **btn_kwargs)

        self.auto_reset_button = ttk.Button(self.auto, text="Reset", command=self.restart)
        self.auto_reset_button.grid(row=6, column=0, columnspan=3, **btn_kwargs)

        self.auto_buttons = [self.auto_er_button, self.auto_hio_button, self.auto_sw_button, self.auto_twin_button,
                             self.auto_blur_button, self.auto_reset_button]

        self.controls.add(self.auto, text="Auto")
        self.controls.add(self.manual, text="Manual")

        # Parameter controls ##########################################################################################
        self.params = ttk.LabelFrame(self.root, text="Parameters", borderwidth=2)
        self.params.grid(row=2, column=1, sticky=tk.S)

        ttk.Label(self.params, text="Shrinkwrap", justify=tk.CENTER).grid(row=0, column=0, columnspan=2)

        ttk.Label(self.params, text="Sigma:", justify=tk.RIGHT).grid(row=1, column=0, sticky=tk.E)
        self.sw_sigma_var = tk.DoubleVar(value=2.0)
        self.sw_sigma_input = ttk.Entry(self.params, textvariable=self.sw_sigma_var, width=6)
        self.sw_sigma_input.grid(row=1, column=1, **btn_kwargs)

        ttk.Label(self.params, text="Threshold:", justify=tk.RIGHT).grid(row=2, column=0, sticky=tk.E)
        self.sw_thresh_var = tk.DoubleVar(value=0.1)
        self.sw_thresh_input = ttk.Entry(self.params, textvariable=self.sw_thresh_var, width=6)
        self.sw_thresh_input.grid(row=2, column=1, **btn_kwargs)

        ttk.Separator(self.params, orient="horizontal").grid(row=3, **sep_kwargs)

        ttk.Label(self.params, text="HIO", justify=tk.CENTER).grid(row=4, column=0, columnspan=2)

        ttk.Label(self.params, text="Beta:", justify=tk.RIGHT).grid(row=5, column=0, sticky=tk.E)
        self.hio_beta_var = tk.DoubleVar(value=0.9)
        self.hio_beta_input = ttk.Entry(self.params, textvariable=self.hio_beta_var, width=6)
        self.hio_beta_input.grid(row=5, column=1, **btn_kwargs)

        ttk.Separator(self.params, orient="horizontal").grid(row=6, **sep_kwargs)

        self.clock = time.perf_counter()

        self.to_direct()
        self.root.mainloop()

    def update_images(self):
        if self.fourier:
            self.img_left = ut.amp_to_photo_image(np.log(np.abs(self.solver.fs_image) + 1))
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
        if 0 < self.hio_beta_var.get() < 1:
            self.solver.hio_constraint(self.hio_beta_var.get())
        else:
            showerror("Error", "HIO failed one or more checks:\n0 < beta < 1")
            return -1
        self.update_images()

    def do_sw(self):
        if (0 < self.sw_sigma_var.get() < 20) and (0 < self.sw_thresh_var.get() < 1):
            self.solver.shrinkwrap(self.sw_sigma_var.get(), self.sw_thresh_var.get())
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
        for button in [self.er_button, self.sw_button, self.fft_button, self.hio_button]:
            button.state(["disabled"])
        for button in [self.ifft_button, self.modulus_button]:
            button.state(["!disabled"])
        self.solver.fft()
        self.update_images()
        pass

    def to_direct(self):
        self.fourier = False
        for button in [self.er_button, self.sw_button, self.fft_button, self.hio_button]:
            button.state(["!disabled"])
        for button in [self.ifft_button, self.modulus_button]:
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
        if not (0 < self.hio_beta_var.get() < 1):
            showerror("Error", "HIO failed one or more checks:\n0 < beta < 1")
            return
        for button in self.auto_buttons + self.ds_buttons + self.fs_buttons:
            button.state(["disabled"])
        self.iterate_hio(self.hio_beta_var.get(), self.num_hio.get())

    def remove_twin(self):
        self.solver.remove_twin()
        self.update_images()

    def gaussian_blur(self):
        self.solver.gaussian_blur()
        self.update_images()

    def restart(self):
        self.solver.reset()
        self.update_images()
        pass


if __name__ == "__main__":
    App(400)
