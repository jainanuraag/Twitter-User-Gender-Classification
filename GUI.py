import tkinter as tk
import Classifiers as cl

HEIGHT = 400
WIDTH = 400


# function being called by button
# to be used by bayes classifier to classify entry text
def button_press(entry):
    ans['text'] = cl.classify(entry)


def button1_press(entry):
    sklearn_ans['text'] = cl.sklearn_MNB_predict(entry)


def button2_press():
    accuracy_label['text'] = cl.validate()


# main window frame
root = tk.Tk()
root.title("Gender Prediction")
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

# Background twitter image for window
bg = tk.PhotoImage(file='BG.png')
bg_label = tk.Label(root, image=bg)
bg_label.place(relwidth=1, relheight=1)

# Background frame for textbox and button
frame = tk.Frame(root, bg='#0084b4', bd=7)
frame.place(relx=0.5, rely=0.01, relwidth=0.75, relheight=0.1, anchor='n')

# Textbox for input of your own tweet
entry = tk.Entry(frame)
entry.place(relwidth=0.68, relheight=1)

# Button to submit tweet
button = tk.Button(frame, text="Submit tweet", fg='#0084b4', command=lambda: button_press(entry.get()))
button.place(relx=.5, relheight=1, relwidth=.12)

# Button to show sklearn's prediction.
button1 = tk.Button(frame, text="Show sklearn prediction", command=lambda: button1_press(entry.get()))
button1.place(relx=.63, relheight=1, relwidth=.2)

# Button to show our implementation's accuracy.
button2 = tk.Button(frame, text="Validation Data Accuracy", command=lambda: button2_press())
button2.place(relx=.85, relheight=1, relwidth=.2)

# Background frame for answer/analysis of user input
answer_frame = tk.Frame(root, bg='#0084b4', bd=7)
answer_frame.place(relx=.5, rely=.38, relwidth=.5, relheight=.25, anchor='n')

# Text for answer/analysis of user input, will change after button press
# #c0deed
ans = tk.Label(answer_frame, bg='#c0deed')
ans.place(relx=.1, relheight=1, relwidth=.3)

# Text for showing prediction using other Native Bayes implementation, for example the sklearn package built-in
# MultibinomialNB.
sklearn_ans = tk.Label(answer_frame, bg='#c0deed')
sklearn_ans.place(relx=.4, relheight=1, relwidth=.3)

# Text for showing the accuracy of our implementation of Native Bayes Model.
accuracy_label = tk.Label(answer_frame, bg='#c0deed')
accuracy_label.place(relx=.7, relheight=1, relwidth=.3)

root.mainloop()
