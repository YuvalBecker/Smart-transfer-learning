import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

class ImageViewer(tk.Frame):
    def __init__(self, master, images):
        super().__init__(master)
        self.images = images
        self.current_image = 0

        # Create a figure and a canvas to display the image
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()

        # Create buttons to switch between images
        self.button_frame = tk.Frame(self)
        self.prev_button = tk.Button(self.button_frame, text="Previous",
                                     command=self.prev_image)
        self.next_button = tk.Button(self.button_frame, text="Next",
                                     command=self.next_image)
        self.number_buttons = []
        for i in range(1, len(self.images) + 1):
            button = tk.Button(self.button_frame, text=str(i),
                               command=lambda i=i: self.show_image(i-1))
            self.number_buttons.append(button)

        # Add a toolbar for zooming
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)

        # Pack the widgets into the frame
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar.update()
        self.button_frame.pack(side=tk.BOTTOM)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button.pack(side=tk.LEFT)
        for button in self.number_buttons:
            button.pack(side=tk.LEFT)

        self.show_image(0)

    def show_image(self, i):
        self.current_image = i
        self.axes.imshow(self.images[i])
        self.canvas.draw()

    def prev_image(self):
        self.show_image((self.current_image - 1) % len(self.images))

    def next_image(self):
        self.show_image((self.current_image + 1) % len(self.images))

# Test the ImageViewer with a sample list of images
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate a list of sample images
    images = []
    for i in range(9):
        img = np.random.rand(50, 50)
        images.append(img)


    import tkinter as tk

    # Create the main window
    root = tk.Tk()
    root.title("Image Viewer")

    # Create a list of sample images

    # Create an instance of the ImageViewer class and specify the main window as the master
    viewer = ImageViewer(master=root, images=images)
    viewer.pack()

# Start the event loop
root.mainloop()
gui = ImageViewer(images=images)
    # Create the GUI and start the event loop
    #show_images(images_list)