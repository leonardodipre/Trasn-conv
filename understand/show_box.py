from PIL import Image, ImageDraw

# Load the image
image_path = r'E:\REFcoco dataset\refcocog\images\COCO_train2014_000000524291_16521.jpg'
image_path = image_path.split("_")[0] + "_" + image_path.split("_")[1] + "_" + image_path.split("_")[2] + ".jpg"

image = Image.open(image_path)

# Define the coordinates of the two boxes
box1 = [ 0.0000, 231.7800, 285.8300, 109.9400]
#box2 = [288.2601, 150.3590, 400.0191, 200.4614]
box2 = [247.6502, 132.4887, 275.4772, 183.2540]

# Create a drawing context
draw = ImageDraw.Draw(image)

# Draw the rectangles on the image
#draw.rectangle(box1, outline="red")
draw.rectangle(box2, outline="blue")

# Display the image
image.show()

