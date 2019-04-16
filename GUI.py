import tkinter as tk

HEIGHT = 400
WIDTH = 400

# function being called by button
# to be used by bayes classifier to try to classify entry text
# and change the label text to the classification
def button_press(entry):
    ans['text'] = "Your input is: " + entry

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
button = tk.Button(frame, text="Submit tweet", fg='#0084b4', command= lambda: button_press(entry.get()))
button.place(relx=0.7, relheight=1, relwidth=0.3)

# Background frame for answer/analysis of user input
answer_frame = tk.Frame(root, bg='#0084b4', bd=7)
answer_frame.place(relx=0.5, rely=0.38, relwidth = 0.5, relheight=0.25, anchor='n')

# Text for answer/analysis of user input, will change after button press
# #c0deed
ans = tk.Label(answer_frame, bg='#c0deed')
ans.place(relheight=1, relwidth=1)

root.mainloop()