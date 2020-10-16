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


        # Creates the structure for the GUI with the title
        self.__window = Tk()
        self.__window.title('Stitch')

        # Creates label for select ImageJ.exe prompt
        self.__s_ij_prompt = Label(self.__window,
                                          text='Select ImageJ.exe file:') \
            .grid(row=3, column=1)

        # Creates the browse button for getting the ImageJ.exe path
        Button(self.__window, text='Browse', command=self.retrieve_ijfolder) \
            .grid(row=3, column=2)

        # Creates the variable label for ImageJ path text
        self.__imgj_path = StringVar()
        self.__selectij = Label(self.__window, text=self.__imgj_path.get(),
                                 bg='white', bd=2,
                                 textvariable=self.__imgj_path, relief='sunken')
        self.__selectij.grid(row=3, column=3, columnspan=3, sticky=W)

        # Creates label for select folder prompt
        self.__s_dir_prompt = Label(self.__window,
                                          text='Select root folder:') \
            .grid(row=5, column=1)

        # Creates the browse button for getting the root folder
        Button(self.__window, text='Browse', command=self.retrieve_rfolder) \
            .grid(row=5, column=2)

        # Creates the variable label for root folder text
        self.__rfolder = StringVar()
        self.__selectDir = Label(self.__window, text=self.__rfolder.get(),
                                 bg='white', bd=2,
                                 textvariable=self.__rfolder, relief='sunken')
        self.__selectDir.grid(row=5, column=3, columnspan=3, sticky=W)

        # Creates check button for using companion.ome file only for stitching
        self.__cb_ome_v = IntVar()
        self.__cb_ome_v.set(0)
        self.__cb_ome = Checkbutton(self.__window,
                                 text='Only use companion.ome file for stitching?',
                                 variable=self.__cb_ome_v, command=self.ome_checked)
        self.__cb_ome.grid(row=6, column=1, sticky=W, columnspan=3)

        # Creates check button for fused_orig creation yes/no
        self.__cb1_var1 = IntVar()
        self.__cb1_var1.set(0)
        self.__cb1 = Checkbutton(self.__window, text='Create stitched tiff using original positions?',
                                variable=self.__cb1_var1)
        self.__cb1.grid(row=7, column=1, sticky=W, columnspan=3)

        # Creates check button for imagej macro run yes/no
        self.__cb2_var1 = IntVar()
        self.__cb2_var1.set(0)
        self.__cb2 = Checkbutton(self.__window,
                                 text='Run imageJ macro?',
                                 variable=self.__cb2_var1)
        self.__cb2.grid(row=8, column=1, sticky=W)

        # Creates the multiplier entry input field

        self.__multi_prompt = Label(self.__window,
                                    text='Enter positions multiplier:') \
            .grid(row=9, column=1)

        self.__multi_input = Entry(self.__window, width=5)
        self.__multi_input.grid(row=9, column=2, padx=5, ipadx=5)

        # Creates the label for errors in multiplier input

        self.__multi_error = Label(self.__window, text='')
        self.__multi_error.grid(row=10, column=1)

        self.__multi_errortxt = 'Multiplier input must be greater than 0!'

        # Creates the run button for running the simulator
        Button(self.__window, text='Run', command=self.stitch_away) \
            .grid(row=11, column=1, sticky=E)

        # Creates button for quitting the stitcher
        Button(self.__window, text='Quit', command=self.quit_func) \
            .grid(row=11, column=2, sticky=W)

        # Adds the Stitch image
        Img = PhotoImage(file=self.__img_file_loc)
        Img = Img.subsample(5)
        imglabel = Label(self.__window, image=Img)
        imglabel.image = Img
        imglabel.grid(row=6, column=4, rowspan=6)

    def retrieve_ijfolder(self):

        selected_path = filedialog.askopenfilename()
        self.__imgj_path.set(selected_path)

    def retrieve_rfolder(self):

        selected_directory = filedialog.askdirectory()
        self.__rfolder.set(selected_directory)

        if not selected_directory == '':
            logging.basicConfig(filename='%s/stitch.log' % selected_directory,
                                format='%(asctime)s %(levelname)-8s %(message)s',
                                level=logging.INFO, datefmt='%d-%m-%Y %H:%M:%S')

    def ome_checked(self):

        if self.__cb_ome_v.get() == 1:
            self.__cb1.config(state=DISABLED)
            self.__cb1_var1.set(0)
            self.__multi_input.delete(0, "end")
            self.__multi_input.config(state=DISABLED)
        else:
            self.__cb1.config(state=NORMAL)
            self.__multi_input.config(state=NORMAL)
            
    def prompt_creator(self):

        prompt_items = [str(self.__rfolder.get()), str(self.__cb1_var1.get()),
                        str(self.__cb2_var1.get()),
                        str(self.__multi_input.get()),
                        str(self.__cb_ome_v.get())]

        prompt = ' "root_dir_path=\'{p[0]}\',y_orig=\'{p[1]}\'' \
                 ',y_macro=\'{p[2]}\',multiplier=\'{p[3]}\'' \
                 ',ome_only=\'{p[4]}\'"'\
            .format(p=prompt_items)

        lab_prompt = self.__imgj_path.get() + " --ij2 --headless --console --run " + \
                     self.__py_file_loc + prompt

        return lab_prompt

    def stitch_away(self):

        lab_prompt = self.prompt_creator()

        if self.__rfolder.get() == '' or self.__imgj_path.get() == '':
            from tkinter import messagebox

            messagebox.showinfo("Warning", "File directory or ImageJ path not"
                                           " selected!")
        else:
            try:
                if self.__cb_ome_v.get() == 1:
                    try:
                        self.__window.destroy()
                        subprocess.call(lab_prompt, shell=True)
                    except Exception as e:
                        logging.exception(str(e))

                elif float(self.__multi_input.get()) > 0:
                    try:
                        self.__window.destroy()
                        subprocess.call(lab_prompt, shell=True)
                    except Exception as e:
                        logging.exception(str(e))

                else:
                    self.__multi_error.configure(text=self.__multi_errortxt, fg='red')
            except ValueError:
                self.__multi_error.configure(text=self.__multi_errortxt, fg='red')

    def quit_func(self):

        self.__window.destroy()

    def start(self):

        self.__window.mainloop()


def main():

    ui = Stitch()
    ui.start()

main()
