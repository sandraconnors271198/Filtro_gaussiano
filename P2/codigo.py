import matplotlib.pyplot as plt
from PIL import Image, ImageFilter ,ImageColor
from sklearn.cluster import KMeans
from collections import Counter
import os
import cv2

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color



def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900, 600), interpolation = cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img

def color_analysis(img,carpeta,num):
    clf = KMeans(n_clusters = 5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    plt.figure(figsize = (12, 8))
    plt.bar(hex_colors,counts.values(), color = hex_colors, label= 'colores')
    plt.savefig(os.getcwd()+'\I_'+carpeta+"\Histograma\Histograma_imagen"+str(num)+".png")
    #plt.show()
    print("Colores: ")
    print(hex_colors)
    return hex_colors

def color_analysis_p(img,dir):
    clf = KMeans(n_clusters = 5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    plt.figure(figsize = (12, 8))
    plt.bar(hex_colors,counts.values(), color = hex_colors, label= 'colores')
    plt.savefig(dir+".png")
    #plt.show()
    print("Colores: ")
    print(hex_colors)
  

def entrenamiento():
	n=[1,2]
	print("Desplegando imagenes de la base de conocimiento llamado aerea:")
	for i in n:
		try:
			image = Image.open(os.getcwd()+"\I_aerea\Entrenamiento"+str(i)+".png") 
		except:
			print("Hubo un error en el archivo cuya ruta es: "+os.getcwd()+"\I_aerea\Entrenamiento"+str(i)+".png")
			return
		image = image.filter(ImageFilter.GaussianBlur)   
		#image.show()
		rgb_im = image.convert('RGB')
		rgb_im.save(os.getcwd()+"\I_aerea\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg" ,"JPEG")

		imagecv2 = cv2.imread(os.getcwd()+"\I_aerea\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg")
		imagecv2 = cv2.cvtColor(imagecv2 , cv2.COLOR_BGR2RGB)
		modified_image = prep_image(imagecv2)
		aerea_colores.append(color_analysis(modified_image,"aerea",i))
		


	n=[1,2,3,4]
	print("Desplegando imagenes de la base de conocimiento llamado comida:")
	for i in n:
		try:
			image = Image.open(os.getcwd()+"\I_comida\Entrenamiento"+str(i)+".jpg") 
		except:
			print("Hubo un error en el archivo cuya ruta es: "+os.getcwd()+"\I_comida\Entrenamiento"+str(i)+".jpg")
			return
		image = image.filter(ImageFilter.GaussianBlur)   
		#image.show()
		rgb_im = image.convert('RGB')
		rgb_im.save(os.getcwd()+"\I_comida\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg" ,"JPEG")

		imagecv2 = cv2.imread(os.getcwd()+"\I_comida\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg")
		imagecv2 = cv2.cvtColor(imagecv2 , cv2.COLOR_BGR2RGB)
		modified_image = prep_image(imagecv2)
		comida_colores.append(color_analysis(modified_image,"comida",i))
		

	n=[1,2,3,4,5,6]
	print("Desplegando imagenes de la base de conocimiento llamado monos:")
	for i in n:
		try:
			image = Image.open(os.getcwd()+"\I_monos\Entrenamiento"+str(i)+".jpg") 
		except:
			print("Hubo un error en el archivo cuya ruta es: "+os.getcwd()+"\I_monos\Entrenamiento"+str(i)+".jpg")
			return
		image = image.filter(ImageFilter.GaussianBlur)   
		#image.show()
		rgb_im = image.convert('RGB')
		rgb_im.save(os.getcwd()+"\I_monos\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg" ,"JPEG")

		imagecv2 = cv2.imread(os.getcwd()+"\I_monos\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg")
		imagecv2 = cv2.cvtColor(imagecv2 , cv2.COLOR_BGR2RGB)
		modified_image = prep_image(imagecv2)
		mono_colores.append(color_analysis(modified_image,"monos",i))


def Prueba():
	#Aquí se va a ir comparando con los colores frecuentes que se extrajeron de las imagenes de conocimiento
	#Carpeta para guardar las imagenes con los colores comunes de los arreglos
	coincidencias_mono=0
	coincidencias_aerea=0
	coincidencias_comida=0
	print("Ponga la dirección de la imagen prueba con su extensión (ya está siendo considerada la dirección actual):")
	dir=input()
	try:
		image = Image.open(os.getcwd()+"\I_"+dir) 
	except:
		print("Hubo un error en el archivo cuya ruta es: "+os.getcwd()+dir)
		return
	image = image.filter(ImageFilter.GaussianBlur)   
	rgb_im = image.convert('RGB')
	rgb_im.save(os.getcwd()+"\Prueba_colores\Imagen_Filtrada.jpg" ,"JPEG")

	img = cv2.imread(os.getcwd()+"\Prueba_colores\Imagen_Filtrada.jpg")
	cv2.imshow('Imagen',img)
	cv2.waitKey(800)
	imagecv2 = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
	modified_image = prep_image(imagecv2)
	color_analysis_p(modified_image,os.getcwd()+"\Prueba_colores\Histograma.jpg")
	prueba.append(hex_colors)

	print("Buscando coincidencias con categoría mono:")
	for i in range(len(mono_colores)):
		for w in range(len(mono_colores[i])):
			print('Color a comparar: '+mono_colores[i][w])
			for n in range(len(prueba)):
				if(mono_colores[i][w]==prueba[n]):
					coincidencias_mono=coincidencias_mono+1

	print("Buscando coincidencias con categoría comida:")
	for i in range(len(comida_colores)):
		for w in range(len(comida_colores[i])):
			print('Color a comparar: '+comida_colores[i][w])
			for n in range(len(prueba)):
				if(comida_colores[i][w]==prueba[n]):
					coincidencias_comida=coincidencias_comida+1


	print("Buscando coincidencias con categoría aerea:")
	for i in range(len(aerea_colores)):
		for w in range(len(aerea_colores[i])):
			print('Color a comparar: '+aerea_colores[i][w])
			for n in range(len(prueba)):
				if(aerea_colores[i][w]==prueba[n]):
					coincidencias_aerea=coincidencias_aerea+1

	
	
	print("coincidencias con categoría mono: "+str(coincidencias_mono))
	print("coincidencias con categoría aerea: "+str(coincidencias_aerea))
	print("coincidencias con categoría comida: "+str(coincidencias_comida))

def main():
	entrenamiento()
	Prueba()

#variables globales
hex_colors =[]
mono_colores=[]
comida_colores=[]
aerea_colores=[]
prueba=[]
main()