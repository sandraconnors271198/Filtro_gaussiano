import matplotlib.pyplot as plt
from PIL import Image, ImageFilter 
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

def color_analysis(img):
    clf = KMeans(n_clusters = 5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    plt.figure(figsize = (12, 8))
    plt.bar(hex_colors,counts.values(), color = hex_colors, label= 'colores')
    plt.savefig("color_analysis_bar.png")
    plt.show()
    print("Colores: ")
    print(hex_colors)

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
		rgb_im = image.convert('RGB')
		rgb_im.save(os.getcwd()+"\i_aerea\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg" ,"JPEG")

		imagecv2 = cv2.imread(os.getcwd()+"\i_aerea\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg")
		imagecv2 = cv2.cvtColor(imagecv2 , cv2.COLOR_BGR2RGB)
		modified_image = prep_image(imagecv2)
		color_analysis(modified_image)
		aerea_colores.append(hex_colors)
		


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
		rgb_im = image.convert('RGB')
		rgb_im.save(os.getcwd()+"\comida\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg" ,"JPEG")

		imagecv2 = cv2.imread(os.getcwd()+"\comida\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg")
		imagecv2 = cv2.cvtColor(imagecv2 , cv2.COLOR_BGR2RGB)
		modified_image = prep_image(imagecv2)
		color_analysis(modified_image)
		comida_colores.append(hex_colors)
		

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
		rgb_im = image.convert('RGB')
		rgb_im.save(os.getcwd()+"\monos\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg" ,"JPEG")

		imagecv2 = cv2.imread(os.getcwd()+"\monos\Filtradas\Imagen_filtro_gaussiano"+str(i)+".jpg")
		imagecv2 = cv2.cvtColor(imagecv2 , cv2.COLOR_BGR2RGB)
		modified_image = prep_image(imagecv2)
		color_analysis(modified_image)
		mono_colores.append(hex_colors)
	


def main():
	entrenamiento()

#variables globales
hex_colors =[]
mono_colores=[]
comida_colores=[]
aerea_colores=[]
main()