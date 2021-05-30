from tkinter import *
import tkinter as tk
root = tk.Tk()
root.geometry('500x500')
l = Label(root, text="Health Condition Prediction", fg= "black", font = "Times 40 bold")
l.place(x=450, y=70)


def heart_prediction():
    root.destroy()
    import heart_program
def kidney_prediction():
    root.destroy()
    import kidney_program
def lung_prediction():
    root.destroy()
    import lung_program


button = Button(root,text="Heart Prediction", font = "Times 18 bold", height = 3, width = 30, command=heart_prediction)
button.place(x=100, y=600)

button1 = Button(root,text="Kidney Prediction", font = "Times 18 bold", height = 3, width = 30, command=kidney_prediction)
button1.place(x=600, y=600)

button2 = Button(root,text="Lung Prediction", font = "Times 18 bold", height = 3, width = 30, command=lung_prediction)
button2.place(x=1100, y=600)

root.mainloop()


