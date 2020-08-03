import os
from tkinter import *
from tkinter import filedialog
import logging
import subprocess



class Stitch:

    def __init__(self):

        self.__py_file_dir = os.path.dirname(os.path.realpath(__file__))
        self.__py_file = r"\stitch.py"
        self.__py_file_loc = self.__py_file_dir + self.__py_file
        self.__img_file = r"\stitch.gif"
        self.__img_file_loc = self.__py_file_dir + self.__img_file
        print(self.__py_file_loc)
        print(self.__img_file_loc)

        # Locates the imageJ exe file
        for root, dirs, files in os.walk(r"E:\Ohjelmat\Fiji.app"): # !!! Change path to ImageJ location
            for file in files:
                if file == 'ImageJ-win64.exe':
                    self.__imgj_path = os.path.join(root, file)

        print(self.__imgj_path)


        #Creates the structure for the GUI with the title
        self.__window = Tk()
        self.__window.title('Stitch')

        # Creates label for select folder prompt
        self.__previousSim_prompt = Label(self.__window,
                                          text='Select root folder:') \
            .grid(row=2, column=1)

        # Creates the browse button for getting the root folder
        Button(self.__window, text='Browse', command=self.retrieve_rfolder) \
            .grid(row=3, column=1)

        # Creates the variable label for root folder text
        self.__rfolder = StringVar()
        self.__selectDir = Label(self.__window, text=self.__rfolder.get(),
                                 bg='white', bd=2,
                                 textvariable=self.__rfolder, relief = 'sunken')
        self.__selectDir.grid(row=3, column=2, columnspan=3, sticky=W)

        # Creates check button for fused_orig creation yes/no
        self.__cb1_var1 = IntVar()
        self.__cb1_var1.set(0)
        self.__cb1 = Checkbutton(self.__window, text='Create Fused.orig tiffs?',
                                variable=self.__cb1_var1)
        self.__cb1.grid(row=4, column=1, sticky=W)

        # Creates check button for imagej macro run yes/no
        self.__cb2_var1 = IntVar()
        self.__cb2_var1.set(0)
        self.__cb2 = Checkbutton(self.__window,
                                 text='Run imageJ macro?',
                                 variable=self.__cb2_var1)
        self.__cb2.grid(row=5, column=1, sticky=W)

        # Creates the run button for running the simulator
        Button(self.__window, text='Run', command=self.stitch_away) \
            .grid(row=6, column=1, sticky=E)

        # Creates button for quitting the stitcher
        Button(self.__window, text='Quit', command=self.quit_func) \
            .grid(row=6, column=2, sticky=W)

        # Adds the Stitch image
        Img = PhotoImage(file=self.__img_file_loc)
        Img = Img.subsample(5)
        imglabel = Label(self.__window, image=Img)
        imglabel.image = Img
        imglabel.grid(row=4, column=3, rowspan=6)

    def retrieve_rfolder(self):

        selected_directory = filedialog.askdirectory()
        self.__rfolder.set(selected_directory)

        if not selected_directory == '':
            logging.basicConfig(filename='%s/stitch.log' % selected_directory,
                                format='%(asctime)s %(levelname)-8s %(message)s',
                                level=logging.INFO, datefmt='%d-%m-%Y %H:%M:%S')

    def stitch_away(self):

        prompt1 = ' "root_dir_path=' + "'" + self.__rfolder.get() + "'" + ",y_orig=" + "'" + "True" + "'" + ",y_macro=" + "'" + "True" + "'" + '"'
        prompt2 = ' "root_dir_path=' + "'" + self.__rfolder.get() + "'" + ",y_orig=" + "'" + "True" + "'" + ",y_macro=" + "'" + "False" + "'" + '"'
        prompt3 = ' "root_dir_path=' + "'" + self.__rfolder.get() + "'" + ",y_orig=" + "'" + "False" + "'" + ",y_macro=" + "'" + "True" + "'" + '"'
        prompt4 = ' "root_dir_path=' + "'" + self.__rfolder.get() + "'" + ",y_orig=" + "'" + "False" + "'" + ",y_macro=" + "'" + "False" + "'" + '"'

        print(self.__rfolder.get())

        if not self.__rfolder.get() == '':
            if self.__cb1_var1.get() == 1 and self.__cb2_var1.get() == 1:
                lab_prompt = self.__imgj_path + " --ij2 --headless --console --run " +\
                             self.__py_file_loc + prompt1
            elif self.__cb1_var1.get() == 1 and self.__cb2_var1.get() == 0:
                lab_prompt = self.__imgj_path + " --ij2 --headless --console --run " + \
                             self.__py_file_loc + prompt2
            elif self.__cb1_var1.get() == 0 and self.__cb2_var1.get() == 1:
                lab_prompt = self.__imgj_path + " --ij2 --headless --console --run " + \
                             self.__py_file_loc + prompt3
            elif self.__cb1_var1.get() == 0 and self.__cb2_var1.get() == 0:
                lab_prompt = self.__imgj_path + " --ij2 --headless --console --run " + \
                             self.__py_file_loc + prompt4
            try:
                self.__window.destroy()
                subprocess.call(lab_prompt, shell=True)
            except Exception as e:
                logging.exception(str(e))

        else:
            from tkinter import messagebox

            messagebox.showinfo("Warning", "No directory selected")

    def quit_func(self):

        self.__window.destroy()

    def start(self):

        self.__window.mainloop()


def main():

    ui = Stitch()
    ui.start()

main()
