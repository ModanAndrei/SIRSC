from pathlib import Path

from PIL import Image, ImageDraw


root = Path(__file__).resolve().parents[2]
output = root / "docs" / "screenshots" / "ui_demo.png"
output.parent.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (1200, 720), (229, 237, 240))
draw = ImageDraw.Draw(image)
draw.rectangle((0, 0, 1200, 86), fill=(23, 50, 58))
draw.text((45, 25), "Semne rutiere, vazute clar.", fill=(247, 250, 248))
draw.rectangle((45, 135, 830, 640), fill=(247, 250, 248), outline=(201, 215, 213), width=2)
draw.rectangle((875, 135, 1150, 640), fill=(247, 250, 248), outline=(201, 215, 213), width=2)
draw.rectangle((165, 225, 685, 545), outline=(219, 58, 52), width=5)
draw.text((195, 180), "Imagine primita cu succes", fill=(23, 50, 58))
draw.text((915, 175), "Detector", fill=(23, 50, 58))
draw.text((915, 235), "Model .pt", fill=(83, 101, 107))
draw.text((915, 280), "Prag incredere   0.35", fill=(83, 101, 107))
draw.text((915, 360), "Detectii", fill=(23, 50, 58))
draw.text((915, 410), "Stop                 94%", fill=(219, 58, 52))
image.save(output)
print(output)