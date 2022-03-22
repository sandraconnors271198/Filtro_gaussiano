import matplotlib.pyplot as plt
from PIL import Image, ImageFilter 
import os

def entrenamiento():
	n=[1,2]
	print("Desplegando imagenes de la base de conocimiento llamado aerea:")
	for i in n:
		try:
			image = Image.open(os.getcwd()+"\i_aerea\Entrenamiento"+str(i)+".png") 
		except:
			print("Hubo un error en el archivo cuya ruta es: "+os.getcwd()+"\i_aerea\Entrenamiento"+str(i)+".png")
			return
		image = image.filter(ImageFilter.GaussianBlur)   
		image.show()

	n=[1,2,3,4]
	print("Desplegando imagenes de la base de conocimiento llamado comida:")
	for i in n:
		try:
			image = Image.open(os.getcwd()+"\comida\Entrenamiento"+str(i)+".jpg") 
		except:
			print("Hubo un error en el archivo cuya ruta es: "+os.getcwd()+"\comida\Entrenamiento"+str(i)+".jpg")
			return
		image = image.filter(ImageFilter.GaussianBlur)   
		image.show()

	n=[1,2,3,4,5,6]
	print("Desplegando imagenes de la base de conocimiento llamado monos:")
	for i in n:
		try:
			image = Image.open(os.getcwd()+"\monos\Entrenamiento"+str(i)+".jpg") 
		except:
			print("Hubo un error en el archivo cuya ruta es: "+os.getcwd()+"\monos\Entrenamiento"+str(i)+".jpg")
			return
		image = image.filter(ImageFilter.GaussianBlur)   
		image.show()


def main():
	entrenamiento()

main()