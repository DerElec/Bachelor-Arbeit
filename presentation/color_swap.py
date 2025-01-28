import fitz
from PIL import Image

input_path = r"D:\Daten\Uni\Bachelor-Arbeit\presentation\delta_one-2.pdf"
output_path = r"D:\Daten\Uni\Bachelor-Arbeit\presentation\output.pdf"

doc = fitz.open(input_path)

# Alle Seiten rendern und bearbeiten
for page_number in range(len(doc)):
    page = doc[page_number]
    pix = page.get_pixmap()  # Render die Seite als Bild
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Farben ändern
    lower_bound = (0, 0, 220)
    upper_bound = (0, 0, 255)
    replacement_color = (0, 255, 255)

    def replace_colors(image, lower, upper, new_color):
        pixels = image.load()
        for y in range(image.height):
            for x in range(image.width):
                r, g, b = pixels[x, y]
                if lower[0] <= r <= upper[0] and lower[1] <= g <= upper[1] and lower[2] <= b <= upper[2]:
                    pixels[x, y] = new_color
        return image

    modified_image = replace_colors(image, lower_bound, upper_bound, replacement_color)
    modified_image.save(f"page_{page_number}.png")  # Testweise speichern

# PDF aus den Bildern neu erstellen
image_list = [f"page_{i}.png" for i in range(len(doc))]
images = [Image.open(img) for img in image_list]
images[0].save(output_path, save_all=True, append_images=images[1:])

print(f"Farbänderung abgeschlossen. Datei gespeichert unter: {output_path}")
