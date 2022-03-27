import matplotlib.pyplot as plt
from PIL import Image, ImageFilter ,ImageColor
from sklearn.cluster import KMeans
from collections import Counter
import os
import cv2
import numpy as np

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
    hex_colors = [ordered_colors[i] for i in counts.keys()]
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
    hex_colors = [ordered_colors[i] for i in counts.keys()]
    return hex_colors
  

def entrenamiento():
	n=[1,2]
	print("******************************Iniciando aprendizaje de la base de conocimiento************************************************************************************")
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

	print("**************************************************************************************************************************************************************************")


def Prueba():
	#Aquí se va a ir comparando con los colores frecuentes que se extrajeron de las imagenes de conocimiento
	#Carpeta para guardar las imagenes con los colores comunes de los arreglos
	coincidencias_mono=0
	coincidencias_aerea=0
	coincidencias_comida=0
	unos_mono=0
	unos_aerea=0
	unos_comida=0
	mono=0
	comida=0
	aerea=0
	may='N'

	prueba.clear()
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
	prueba.append(color_analysis_p(modified_image,os.getcwd()+"\Prueba_colores\Histograma.jpg"))
	print("**************************************************************************************************************************************************************************")
	print("Buscando coincidencias con categoría mono:")
	for i in range(len(mono_colores)):
		for w in range(len(mono_colores[i])):
			for n in range(len(prueba)):
				for r in range(len(prueba[n])):
						if((abs(mono_colores[i][w][0]-prueba[n][r][0])<10) and (abs(mono_colores[i][w][1]-prueba[n][r][1])<10) and (abs(mono_colores[i][w][2]-prueba[n][r][2])<10)):
							print("Coincidencia entre: "+str(mono_colores[i][w])+ " y "+str(prueba[n][r]))
							coincidencias_mono=coincidencias_mono+1

	print(' ')
	print("Buscando coincidencias con categoría comida:")
	for i in range(len(comida_colores)):
		for w in range(len(comida_colores[i])):
			for n in range(len(prueba)):
				for r in range(len(prueba[n])):
					if((abs(comida_colores[i][w][0]-prueba[n][r][0])<10) and (abs(comida_colores[i][w][1]-prueba[n][r][1])<10) and (abs(comida_colores[i][w][2]-prueba[n][r][2])<10)):
							print("Coincidencia entre: "+str(comida_colores[i][w])+ " y "+str(prueba[n][r]))
							coincidencias_comida=coincidencias_comida+1


	print(' ')
	print("Buscando coincidencias con categoría aerea:")
	for i in range(len(aerea_colores)):
		for w in range(len(aerea_colores[i])):
			for n in range(len(prueba)):
				for r in range(len(prueba[n])):
					if((abs(aerea_colores[i][w][0]-prueba[n][r][0])<10) and (abs(aerea_colores[i][w][1]-prueba[n][r][1])<10) and (abs(aerea_colores[i][w][2]-prueba[n][r][2])<10)):
							print("Coincidencia entre: "+str(aerea_colores[i][w])+ " y "+str(prueba[n][r]))
							coincidencias_aerea=coincidencias_aerea+1


	print("***************************************************************************************************************************************************************************")
	
	print("Coincidencias con categoría mono: "+str(coincidencias_mono) +". Probabilidad de que sea mono según esta prueba es de "+ str(coincidencias_mono/30) +" %")
	print("Coincidencias con categoría aerea: "+str(coincidencias_aerea) +". Probabilidad de que sea una imagen aerea según esta prueba es de "+ str(coincidencias_aerea/10) +" %")
	print("Coincidencias con categoría comida: "+str(coincidencias_comida) +". Probabilidad de que sea comida según esta prueba es de "+ str(coincidencias_comida/20) +" %")

	print("*************************************Decisión tomada según la frecuencia de su histograma**********************************************************************************")

	if((coincidencias_mono/30 > coincidencias_aerea/10 )and(coincidencias_mono/30 > coincidencias_comida/20)):
		print("Es un mono")
		mono=1
	elif(coincidencias_aerea/10 > coincidencias_comida/20):
		print("Es una imagen aerea")
		aerea=1
	else:
		print("Es comida")
		comida=1

	print("**************************************************************************************************************************************************************************")

	print("Creando imagenes de máscara de los colores dicriminantes de la imagen: ")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	print("Máscara amarillo")
	amarilloAlto = np.array([219, 200, 90])
	amarilloBajo = np.array([200, 170, 00])
	mask = cv2.inRange(img, amarilloBajo, amarilloAlto)
	image = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
	cv2.imshow("mascara_amarilla", mask)
	cv2.waitKey(1000)
	cv2.imwrite(os.getcwd()+"\Prueba_colores\Mascara_amarillo.jpg",mask)
	unos_comida=np.sum(mask == 255)

	print("Máscara negro")
	negroBajo = np.array([17, 17, 16])
	negroAlto = np.array([35, 44, 50])
	mask = cv2.inRange(img, negroBajo, negroAlto)
	image = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
	cv2.imshow("Mascara_negro", mask)
	cv2.waitKey(800)
	cv2.imwrite(os.getcwd()+"\Prueba_colores\Mascara_negro.jpg",mask)
	unos_mono=np.sum(mask == 255)

	print("Máscara gris")
	grisBajo = np.array([162, 158, 151])
	grisAlto = np.array([183, 180, 178])
	mask = cv2.inRange(img, grisBajo, grisAlto)
	image = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
	cv2.imshow("mascarac_gris", mask)
	cv2.waitKey(800)
	cv2.imwrite(os.getcwd()+"\Prueba_colores\Mascara_gris.jpg",mask)
	unos_aerea=np.sum(mask == 255)
	
	if(unos_mono> unos_aerea and unos_mono>unos_comida):
		mayor='M'
	elif(unos_aerea>unos_comida):
		mayor='A'
	else:
		mayor='C'
	print(' ')
	print("************************************Decisión tomada según su máscara y el histograma************************************************************************************")

	if(mayor=='A' and aerea==1):
		print("Es una imagen aerea")

	elif(mayor=='M' and mono==1):
		print("Es una imagen de un mono")

	elif(mayor=='C' and comida==1):
		print("Es una imagen de comida")

	else:
		print("No lo tengo claro")

	print("**************************************************************************************************************************************************************************")



	
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