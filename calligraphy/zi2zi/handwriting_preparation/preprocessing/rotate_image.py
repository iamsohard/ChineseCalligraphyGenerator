import pdb

from PIL import Image
img = Image.open("../images/Lin.jpg")

img2 = img.rotate(10)
img.rotate(-3).save("../images/Lin.jpg",quality=100)
pdb.set_trace()
