import matplotlib.pyplot as mt
import imageio
import numpy as np
import scipy.special as sks
import scipy.misc
from PIL import Image, ImageDraw
from PIL import ImageGrab
import os
import tkinter as tk
from PIL import ImageTk
import math as m
# img = Image.open("C:/Users/Asus/Desktop/dva.jpg")
# img = img.save('dva.png')
# img = Image.open("C:/Users/Asus/Desktop/dva.png")
# resized_img = img.resize((28, 28), Image.ANTIALIAS)
# resized_img.save('dvares.png')
# a = np.zeros([3,2])
# a[0,0] = 1
# a[0,1] = 2
# a[1,0] = 9
# a[2,1] = 12
# mt.imshow(a, interpolation = "nearest")
# input()
outfileih = "ih.npy"
outfileho = "ho.npy"
loadedih = np.load(outfileih)
loadedho = np.load(outfileho)

class NN:
	def __init__(self, inputnodes, hiddennodes, outputnoddes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnoddes

		self.lr = learningrate

		self.wih = np.random.normal(0.0, pow (self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		# self.who = loadedho
		# self.wih = loadedih

		self.activation_function = lambda x: sks.expit(x)

		self.inverse_activation_function = lambda x: sks.logit(x)

		pass



	def train(self, inputs_list, target_list):
		#преобразовать список входных значений в двумерный массив
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(target_list, ndmin=2).T

		#рассчитать входящие сигналы для скрытого слоя
		hidden_inputs = np.dot(self.wih, inputs)
		# рассчитать выходящие сигналы для скрытого слоя
		hidden_outputs = self.activation_function(hidden_inputs)

		# рассчитать входящие сигналы для выходного слоя
		final_inputs = np.dot(self.who, hidden_outputs)
		# расчитать выходящие сигналы для выходного слоя
		final_outputs = self.activation_function(final_inputs)

		# расчитываем ошибку
		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)
		input_errors = np.dot(self.wih.T, hidden_errors)

		# корректировка весов между скрытым и выходным
		self.who+= self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

		# корректировка весов между входным и скрытым
		self.wih+= self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))









		# pass


	def query(self, inputs_list):
		inputs = np.array(inputs_list, ndmin = 2).T


		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

	def backquery(self, targets_list):
		# transpose the targets list to a vertical array
		final_outputs = np.array(targets_list, ndmin=2).T

		# calculate the signal into the final output layer
		final_inputs = self.inverse_activation_function(final_outputs)

		# calculate the signal out of the hidden layer
		hidden_outputs = np.dot(self.who.T, final_inputs)
		# scale them back to 0.01 to .99
		hidden_outputs -= np.min(hidden_outputs)
		hidden_outputs /= np.max(hidden_outputs)
		hidden_outputs *= 0.98
		hidden_outputs += 0.01

		# calculate the signal into the hidden layer
		hidden_inputs = self.inverse_activation_function(hidden_outputs)

		# calculate the signal out of the input layer
		inputs = np.dot(self.wih.T, hidden_inputs)
		# scale them back to 0.01 to .99
		inputs -= np.min(inputs)
		inputs /= np.max(inputs)
		inputs *= 0.98
		inputs += 0.01

		return inputs







input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Training------------------------------------------------------------------|
def trainer():
	training_data_file = open("C:/Users/фыы/Desktop/Desk/mnist_train.csv", 'r')
	training_data_list = training_data_file.readlines()
	training_data_file.close()

	epoches = 1
	a = 0.0000
	for e in range (epoches):
		for record in training_data_list:
			all_values = record.split(',')
			inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			targets = np.zeros(output_nodes)+0.01
			targets[int(all_values[0])] = 0.99
			n.train(inputs, targets)
			a+=0.0016
			print(round(a, 2), "%")
			tr_text = str(round(a, 2))+"%"
			lab_tr.configure(text = tr_text)


			pass
		pass
# ------------------------------------------------------------------------------------|

# Test-------------------------------------------------------|
def tester():
	test_data_file = open("C:/Users/фыы/Desktop/Desk/mnist_test.csv", 'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()

	scorecard = []
	for test_record in test_data_list:
		test_values = test_record.split(',')
		correct_label = int(test_values[0])
		#print(correct_label, "истинный маркер")
		inputs = (np.asfarray(test_values[1:]) / 255.0 * 0.99) + 0.01

		outputs = n.query(inputs)
		label = np.argmax(outputs)
		#print(label, "ответ сети")

		if (label == correct_label):
			scorecard.append(1)
		else:
			scorecard.append(0)
			pass
		pass
	scorecard_array = np.asarray(scorecard)
	txt_tst = "Эффективность = "+ str(scorecard_array.sum()/scorecard_array.size)
	print("Эффективность = ", scorecard_array.sum()/scorecard_array.size)

def owntester():

	image1.save("dvares.png")
	img_array = Image.open("dvares.png")
	mt.imshow(img_array, cmap = 'Greys', interpolation = 'None')
	resized_img = img_array.resize((28, 28), Image.ANTIALIAS)
	resized_img.save("dvares.png")
	img_array = imageio.imread("dvares.png")

	img_data = 255.0 - img_array.reshape(784)
	img_data = (img_data / 255.0 * 0.99) + 0.01


	outputs = n.query(img_data)
	label = np.argmax(outputs)
	#math.ceil(v*100)/100
	answ = int(outputs[label]*10000)/10000
	#answ = m.round(answ, 3)
	#answ = np.double(answ)
	txtlab = "Ответ сети: " + str(label) + "\n" + "Уверенность: "+ str(answ*100)+"%"
	lab_o.config(text =txtlab)
	test_image_array = np.asfarray(img_data).reshape((28,28))
	mt.imshow(test_image_array, cmap = 'Greys', interpolation = 'None')
	#mt.show()
	backtester(label)
	#mt.show()


# qur = n.query((np.asfarray(test_values[1:]) / 255.0 * 0.99) + 0.01)

# -------------------------------------------------------------------------------------|
# -------------------------------------------------------------------------------------|
def backtester(label):
	# label = 0
	# create the output signals for this label
	targets = np.zeros(output_nodes) + 0.01
	# all_values[0] is the target label for this record
	targets[label] = 0.99
	# targets[2] = 0.99


	# get image data
	image_data = n.backquery(targets)

	# plot image data
	mt.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')









def Load():
	lab_load['text'] = "Готово"
	n.who = loadedho
	n.wih = loadedih
def O():

	owntester()
def tr():
	trainer()
	np.save(outfileih, n.wih)
	np.save(outfileho, n.who)
def tst():
	tester()

def udal():
	global draw
	global image1
	del draw
	del image1
	image1 = Image.new("RGB", (200, 200), (255,255,255))

	can.delete("all")
	lab_o['text'] = ""
	draw = ImageDraw.Draw(image1)



brush_size = 7
def dr(event):
	global brush_size


	x1 = event.x - brush_size
	x2 = event.x + brush_size
	y1 = event.y - brush_size
	y2 = event.y + brush_size
	can.create_oval(x1, y1, x2, y2, fill = "black", outline = "black")

	draw.line([x1, y1, x2, y2],fill="black", width = 7)

image1 = Image.new("RGB", (200, 200), (255,255,255))
draw = ImageDraw.Draw(image1)
text_load = ""
root = tk.Tk()

root.title("Neuro")
root.geometry("650x450+300+200")

can = tk.Canvas(root, width = 28, height = 28, bg = "grey")

can.bind("<B1-Motion>", dr)

lab_load = tk.Label(root)
lab_load.configure(text = "")

lab_descr = tk.Label(root)
lab_descr.configure(text = "Рисуйте цифры в середине окна")


btn_load = tk.Button(root, height = 3, width = 20, text = "Загрузить память", command = Load, bg = '#ffffff')

btn_o = tk.Button(root, height = 3, width = 20, text = "Тест", command = O, bg = '#ffffff')
btn_tr = tk.Button(root, height = 3, width = 20, text = "Обучить", command = tr, bg = '#ffffff')

btn_test = tk.Button(root, height = 3, width = 20, text = "Тест", command = tst, bg = '#ffffff')
lab_test = tk.Label(root)
lab_test.configure(width = 20, text = "")

btn_del = tk.Button(root, height = 3, width = 20, text = "Очистить", command = udal, bg = '#ffffff')

lab_o = tk.Label(root)
lab_o.configure(width = 20, text = "")

lab_tr = tk.Label(root)
lab_tr.configure(width = 20, text = "")

btn_load.grid(row = 0, column = 0, columnspan = 1)
lab_load.grid(row = 0, column = 1)
lab_descr.grid(row = 0, column =3, columnspan = 2)
btn_o.grid(row = 1, column = 0, columnspan = 1)
lab_o.grid(row = 1, column = 1, columnspan = 1)

#btn_test.grid(row = 3, column = 0, columnspan = 1)
#lab_tr.grid(row = 3, column = 1, columnspan = 1)


can.grid(row = 2, column = 3)
btn_del.grid(row = 3, column = 3)
root.mainloop()










