"""누운 배우를 하나씩 비스듬히(45°) 가까이 찍어 모음판을 만든다 — 떠 있는지 확인용."""
import sys, time
import numpy as np
import cosysairsim as airsim
from cosysairsim.utils import euler_to_quaternion
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")
c = airsim.VehicleClient(); c.confirmConnection()
names = sorted(n for n in c.simListInstanceSegmentationObjects() if n.startswith("Person_Lying"))
tiles = []
for n in names:
    p = c.simGetObjectPose(n).position
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(p.x_val - 4.0, p.y_val, p.z_val - 4.0), euler_to_quaternion(0, 0, 0)), True)
    c.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), euler_to_quaternion(0, -0.785, 0)))
    time.sleep(0.5)
    r = c.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    img = Image.fromarray(np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)[:, :, ::-1])
    w, h = img.size
    img = img.crop((w // 2 - 450, h // 2 - 300, w // 2 + 450, h // 2 + 300)).resize((300, 200))
    ImageDraw.Draw(img).text((5, 5), n, fill=(255, 0, 0))
    tiles.append(img)
sheet = Image.new("RGB", (300 * 5, 200 * ((len(tiles) + 4) // 5)), (255, 255, 255))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 5) * 300, (i // 5) * 200))
sheet.save(sys.argv[1])
print("done", len(tiles))
