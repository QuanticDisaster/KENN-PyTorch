# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:09:09 2021

@author: Tidop
"""


import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import Frame
import re
import subprocess
import os

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def save(self, email):
        """
        Save the email
        :param email:
        :return:
        """
        try:

            # save the model
            self.model.email = email
            self.model.save()

            # show a success message
            self.view.show_success(f'The email {email} saved!')

        except ValueError as error:
            # show an error message
            self.view.show_error(error)
    
    def update_train_filename(self, filename):
        self.model.train_cloud = filename
    
    def update_test_filename(self, filename):
        self.model.test_cloud = filename
    
    def training_launch(self):
        self.model.training_launch()
    
    def classification_launch(self):
        self.model.classification_launch()
    
    def update_command_label(self, msg):
        self.view.commande_label_msg.set(msg)
    
    def update_instructions_label(self):
        self.view.instructions['text'] = 'Go check .log file in the output folder to see progression ! \nTo interrupt it, close the application'

class View(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        # create widgets
        label = ttk.Label(text="Select mode :")
        label.pack(fill='x', padx=5, pady=5)

        self.r1 = ttk.Radiobutton(parent, text='Training', value='training', command = self.show_training_options )
        self.r1.pack(fill='x', padx=5, pady=5)
        
        self.r2 = ttk.Radiobutton(parent, text='Classification', value='Inference', command= self.show_classification_options)
        self.r2.pack(fill='x', padx=5, pady=5)
        

    def show_training_options(self):
        
        for widget in self.parent.winfo_children():
            widget.destroy()
        
        self.__init__(self.parent)
        
        #create widget to choose train cloud
        training_frame = Frame(self.parent)
        training_frame.columnconfigure(0, weight=3)
        training_frame.columnconfigure(1, weight=1)
        
        label = ttk.Label(training_frame, text="Select training cloud (.txt) :")
        label.grid(column=0, row=0, sticky=tk.W,  padx=5, pady=5)
        
        button = ttk.Button(training_frame, text="...", command=self.button_training_load_train)
        button.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        self.train_label = ttk.Label(training_frame, text=self.controller.model.train_cloud)
        self.train_label.grid(column=0, row=1, columnspan=2,  padx=15, pady=5, sticky=tk.W)

        training_frame.pack(fill='x', pady=5)
        
        #create widget to choose test cloud
        test_frame = Frame(self.parent)
        test_frame.columnconfigure(0, weight=3)
        test_frame.columnconfigure(1, weight=1)
        
        label = ttk.Label(test_frame, text="Select test cloud (.txt) :")
        label.grid(column=0, row=0, sticky=tk.W,  padx=5, pady=5)
        
        button = ttk.Button(test_frame, text="...", command=self.button_training_load_test)
        button.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        
        self.test_label = ttk.Label(test_frame, text=self.controller.model.test_cloud)
        self.test_label.grid(column=0, row=1, columnspan=2, padx=15, pady=5, sticky=tk.W)
        
        test_frame.pack(fill='x', pady=5)        
        
        #widget to enter features index
        frame_features = Frame(self.parent)
        self.feat_index = tk.StringVar('')
        label = tk.Label(frame_features, text="features (optionnal) :").pack(fill='x', padx=5, pady=1, side='left')
        widget_features = ttk.Entry(frame_features, textvariable=self.feat_index, width=30)#, validatecommand=(reg, '%P'), validate='focus')
        widget_features.pack(fill='x', padx=5, pady=1, side='right')
        frame_features.pack(fill='x', padx=5, pady=1)
        
        self.message_feat = ttk.Label(self, text='', foreground='red')
        self.message_feat.pack(fill='x', padx=5, pady=1)
        
        #widget to enter labels index
        frame_labels = Frame(self.parent)
        self.label_index = tk.StringVar('')
        label = tk.Label(frame_labels, text="labels :").pack(fill='x', padx=5, pady=1, side='left')
        widget_labels = ttk.Entry(frame_labels, textvariable=self.label_index, width=30)#, validatecommand=(reg, '%P'), validate='key')
        widget_labels.pack(fill='x', padx=5, pady=1,side='right')
        frame_labels.pack(fill='x', padx=5, pady=1)
        
        
        #widget to choose size
        frame_size = Frame(self.parent)
        self.size_index = tk.StringVar('')
        self.size_index.set("48000")
        size = tk.Label(frame_size, anchor='w', justify='left', text="size : \nLow value reduce GPU memory consumption").pack(fill='x', padx=5, pady=1, side='left')
        widget_size = ttk.Entry(frame_size, textvariable=self.size_index, width=30)#, validatecommand=(reg, '%P'), validate='key')
        widget_size.pack(fill='x', padx=5, pady=1,side='right')
        frame_size.pack(fill='x', padx=5, pady=1)
        
        #widget to choose epoch
        frame_epoch = Frame(self.parent)
        self.epoch_index = tk.StringVar('')
        self.epoch_index.set("300")
        epoch = tk.Label(frame_epoch, anchor='w', justify='left', text="epoch :\nNumber of iterations").pack(fill='x', padx=5, pady=1, side='left')
        widget_epoch = ttk.Entry(frame_epoch, textvariable=self.epoch_index, width=30)#, validatecommand=(reg, '%P'), validate='key')
        widget_epoch.pack(fill='x', padx=5, pady=1,side='right')
        frame_epoch.pack(fill='x', padx=5, pady=1)
        
        #widget to choose output
        frame_output = Frame(self.parent)
        frame_output.columnconfigure(0, weight=3)
        frame_output.columnconfigure(1, weight=1)
        label = tk.Label(frame_output, text="Output folder :")
        label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        
        button = ttk.Button(frame_output, text="...", command=self.button_output)
        button.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        self.output_label = ttk.Label(frame_output, text=self.controller.model.output)
        self.output_label.grid(row=1, sticky=tk.W, padx=5, pady=5,  columnspan=2)
        frame_output.pack(fill='x', padx=5, pady=1)
        
        #widget button to launch training
        button = ttk.Button(self.parent, text="Train", command=self.button_training_launch)
        button.pack( padx=5, pady=15)
        
        #widget to print success or argument malformation
        self.message = ttk.Label(self.parent, text='')
        self.message.pack(fill='x', padx=5, pady=5)
        
        #widget to print the command-line executed
        self.commande_label_msg = tk.StringVar('')
        self.command_label = tk.Label(self.parent, textvariable=self.commande_label_msg, padx=1, wraplength=600)
        self.command_label.pack(fill='x', padx=5, pady=5)
        
        #widget to print to go to the log file
        self.instructions = ttk.Label(self.parent, text='')
        self.instructions.pack(fill='x', padx=5, pady=30)
        
    def show_classification_options(self):
        for widget in self.parent.winfo_children():
            widget.destroy()
            
        self.__init__(self.parent)
        
        #widget to choose model
        model_frame = Frame(self.parent)
        model_frame.columnconfigure(0, weight=3)
        model_frame.columnconfigure(1, weight=1)
        
        label = ttk.Label(model_frame, text="Select model (.pt) :")
        label.grid(column=0, row=0, sticky=tk.W,  padx=5, pady=5)
        
        button = ttk.Button(model_frame, text="...", command=self.button_classification_load_model)
        button.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        self.model_label = ttk.Label(model_frame, text= self.controller.model.model)
        self.model_label.grid(column=0, row=1, columnspan=2, padx=15, pady=5, sticky=tk.W)
    
        model_frame.pack(fill='x', pady=5)
        
        #widget to choose cloud
        cloud_frame = Frame(self.parent)
        cloud_frame.columnconfigure(0, weight=3)
        cloud_frame.columnconfigure(1, weight=1)
        
        label = ttk.Label(cloud_frame, text="Select cloud (.txt) :")
        label.grid(column=0, row=0, sticky=tk.W,  padx=5, pady=5)
        
        button = ttk.Button(cloud_frame, text="...", command=self.button_classification_load_cloud)
        button.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        self.cloud_label = ttk.Label(cloud_frame, text= self.controller.model.cloud)
        self.cloud_label.grid(column=0, row=1, columnspan=2, padx=15, pady=5, sticky=tk.W)
       
        cloud_frame.pack(fill='x', pady=5)

        #widget to enter features index
        frame_features = Frame(self.parent)
        self.feat_index = tk.StringVar('')
        label = tk.Label(frame_features, text="features (optionnal) :").pack(fill='x', padx=5, pady=1, side='left')
        widget_features = ttk.Entry(frame_features, textvariable=self.feat_index, width=30)
        widget_features.pack(fill='x', padx=5, pady=1, side='right')
        frame_features.pack(fill='x', padx=5, pady=1)
        
        #widget to choose size
        frame_size = Frame(self.parent)
        self.size_index = tk.StringVar('')
        self.size_index.set("48000")
        size = tk.Label(frame_size, anchor='w', justify='left', text="size : \nLow value reduce GPU memory consumption").pack(fill='x', padx=5, pady=1, side='left')
        widget_size = ttk.Entry(frame_size, textvariable=self.size_index, width=30)#, validatecommand=(reg, '%P'), validate='key')
        widget_size.pack(fill='x', padx=5, pady=1,side='right')
        frame_size.pack(fill='x', padx=5, pady=1)
        
        #widget to choose output
        frame_output = Frame(self.parent)
        frame_output.columnconfigure(0, weight=3)
        frame_output.columnconfigure(1, weight=1)
        label = tk.Label(frame_output, text="Output folder :")
        label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        
        button = ttk.Button(frame_output, text="...", command=self.button_output)
        button.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        self.output_label = ttk.Label(frame_output, text=self.controller.model.output)
        self.output_label.grid(row=1, sticky=tk.W, padx=5, pady=5,  columnspan=2)
        frame_output.pack(fill='x', padx=5, pady=1)
        
        #widget button to launch classif
        button = ttk.Button(self.parent, text="Classify", command=self.button_classification_launch)
        button.pack( padx=5, pady=15)#.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        
        #widget to print success or argument malformation
        self.message = ttk.Label(self.parent, text='')
        self.message.pack(fill='x', padx=5, pady=5)
        
        
        #widget to print the command-line executed
        self.commande_label_msg = tk.StringVar('')
        self.command_label = tk.Label(self.parent, textvariable=self.commande_label_msg, padx=1, wraplength=600)
        self.command_label.pack(fill='x', padx=5, pady=5)
        
        #widget to print to go to the log file
        self.instructions = ttk.Label(self.parent, text='')
        self.instructions.pack(fill='x', padx=5, pady=30)
        
    def set_controller(self, controller):
        """
        Set the controller
        :param controller:
        :return:
        """
        self.controller = controller
        
    def button_classification_launch(self):
        print('hop')
        try:
            
            self.controller.model.model = self.model_label['text']
            self.controller.model.cloud = self.cloud_label['text']
            self.controller.model.output = self.output_label['text']
            self.controller.model.feat = self.feat_index.get()
            self.controller.model.size = self.size_index.get()
            self.controller.classification_launch()
            self.show_success('Success')
        except ValueError as error:
            print('error')
            self.show_error(error)

    def button_training_load_train(self):
        """
        Handle button click event
        :return:
        """
        #self.controller.model.train_cloud = 
        self.train_label['text'] = fd.askopenfilename()
    
    def button_training_load_test(self):
        """
        Handle button click event
        :return:
        """
        self.test_label['text'] = fd.askopenfilename()

    def button_output(self):
        self.output_label['text'] = fd.askdirectory()
    
    def button_training_launch(self):
        try:
            #Here we get the elements and validate the parameters under the hood
            self.controller.model.train_cloud = self.train_label['text']
            self.controller.model.test_cloud = self.test_label['text']
            self.controller.model.output = self.output_label['text']
            self.controller.model.feat = self.feat_index.get()
            self.controller.model.labels = self.label_index.get()
            self.controller.model.size = self.size_index.get()
            self.controller.model.epoch = self.epoch_index.get()
            self.controller.training_launch()   
            self.show_success('Success')
        
        except ValueError as error:
            self.show_error(error)
    
    def button_classification_load_model(self):
        self.model_label['text'] = fd.askopenfilename()
    
    def button_classification_load_cloud(self):
        self.cloud_label['text'] = fd.askopenfilename()
    
        
    def show_error(self, message):
        """
        Show an error message
        :param message:
        :return:
        """
        self.message['text'] = message
        self.message['foreground'] = 'red'
        self.message.after(3000, self.hide_message)

    def show_success(self, message):
        """
        Show a success message
        :param message:
        :return:
        """
        self.message['text'] = message
        self.message['foreground'] = 'green'
        self.message.after(5000, self.hide_message)

    def hide_message(self):
        """
        Hide the message
        :return:
        """
        self.message['text'] = ''



class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Point Cloud Classification')

        # create a model
        self.model = Model()

        # create a view and place it on the root window
        self.view = View(self)
        #view.grid(row=0, column=0, padx=10, pady=10)

        # create a controller
        self.controller = Controller(self.model, self.view)

        # set the controller to view
        self.view.set_controller(self.controller)
        
        # set the controller to model
        self.model.set_controller(self.controller)
        
        
        
        
        
class Model:
    
    train_cloud =  None
    test_cloud = None
    feat = ''
    labels = ''
    output = None
    __feat = ''
    __labels = ''
    controller = None
    model = None
    cloud = None
    path_script = r''
    path_anaconda = r'env\torch_env_38'
        
    def __init__(self):
        pass
    
    def set_controller(self, controller):
        """
        Set the controller
        :param controller:
        :return:
        """
        self.controller = controller
        
    @property
    def feat(self):
        return self.__feat

    @feat.setter
    def feat(self, value):
        """
        Validate the feature parameter
        :param value:
        :return:
        """
        pattern = r'[0-9]*(,[0-9]*)*'
        if re.fullmatch(pattern, value):
            self.__feat = value
        else:
            raise ValueError(f'Invalid values for features : {value}')
    
    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, value):
        """
        Validate the email
        :param value:
        :return:
        """
        pattern = r'[0-9][0-9]*'
        if re.fullmatch(pattern, value):
            self.__labels = value
        else:
            raise ValueError(f'Invalid values for labels: {value}')
    
    @property
    def size(self):
        return self.__size
    
    @size.setter
    def size(self, value):
        """
        Validate the size parameter
        :param value:
        :return:
        """
        pattern = r'[0-9][0-9]*'
        if re.fullmatch(pattern, value):
            self.__size = value
        else:
            raise ValueError(f'Invalid value for size: {value}')
    
    @property
    def epoch(self):
        return self.__epoch
    
    @size.setter
    def epoch(self, value):
        """
        Validate the epoch parameter
        :param value:
        :return:
        """
        pattern = r'[0-9][0-9]*'
        if re.fullmatch(pattern, value):
            self.__epoch = value
        else:
            raise ValueError(f'Invalid value for epoch: {value}')

    def classification_launch(self):
        output_path = 'output0'
        while os.path.exists( os.path.join(self.output, output_path)):
            output_path = output_path[:6] + str( int( output_path[6] ) + 1 )
        
        os.mkdir(os.path.join(self.output, output_path))
    
        self.command = ('python -u "' + os.path.join(self.path_script, 'inference.py') + '"'
                        ' --model "' + self.model + '"' +
                        ((' --features ' + self.__feat) if self.__feat != '' else '') +
                        ' --input "' + self.cloud + '"'
                        ' --size "' + self.__size + '"'
                        ' --output "' + os.path.join(self.output, output_path, 'cloud_classified.txt') + '"'
                        ' > "' + os.path.join(self.output, output_path, 'output.log') + '"')
        
        full_command = ( os.path.join(self.path_anaconda, r'Scripts\activate.bat') + r' && ' 
                       + self.command)

        self.process = subprocess.Popen( full_command,
                       stdout=subprocess.PIPE, 
                       universal_newlines=True)
        
        self.controller.update_command_label(full_command)
        self.controller.update_instructions_label()
        
    def training_launch(self):
        """
         launch command
        :return:
        """
        output_path = 'output0'
        while os.path.exists( os.path.join(self.output, output_path)):
            output_path = output_path[:6] + str( int( output_path[6] ) + 1 )
        
        os.mkdir(os.path.join(self.output, output_path))
    
        self.command = ('python -u "' + os.path.join(self.path_script,'training.py ') + '"' +
                ' --train "' + self.train_cloud + '"' +
                ' --test "' + self.test_cloud + '"' +
                ((' --features ' + self.__feat) if self.__feat != '' else '') +
                ' --labels ' + self.__labels +
                ' --size ' + self.__size +
                ' --epoch ' + self.__epoch +
                ' --output "' + os.path.join(self.output, output_path) + '"' +
                ' > "' + os.path.join(self.output, output_path, 'output.log') + '"')
             
        #subprocess.run
        full_command = ( os.path.join(self.path_anaconda, r'Scripts\activate.bat') + r' && ' 
                       + self.command )
        print(full_command)
        self.process = subprocess.Popen( full_command,
                       stdout=subprocess.PIPE, 
                       universal_newlines=True)
        
        self.controller.update_command_label(full_command)
        self.controller.update_instructions_label()
        
if __name__ == '__main__':
    app = App()
    app.mainloop()   
    